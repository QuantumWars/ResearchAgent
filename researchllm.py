import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
import random
import requests
import json

# Get API keys from Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]

# Initialize the language model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    max_output_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define the prompt template
RESEARCH_PROMPT = PromptTemplate(
    input_variables=["research_field", "research_topic"],
    template="""
You are an expert research consultant specializing in {research_field}. Your task is to generate research categories and prompts based on a given research topic.

Given research topic: {research_topic}

Guidelines:
1. Create 3-5 distinct research categories related to the topic.
2. For each category, provide:
   a) A brief description of the category
   b) A specific research prompt
   c) Key concepts or terms relevant to this category

Format your response based on these instructions:
- Use bullet points for lists of categories or key concepts.
- Use numbered lists for the main categories.
- Use bold for category titles and research prompts.
- Structure your response with clear sections for better readability.

Constraints:
- Ensure each category is distinct and covers a different aspect of the topic.
- Research prompts should be specific, answerable, and suitable for academic inquiry.
- Avoid overlapping content between categories.

Based on the above guidelines, provide your structured research categories and prompts below:
"""
)

def parse_research_categories(text):
    categories = re.split(r'\*\*\d+\.\s+', text)[1:]
    research_dict = {}
    for category in categories:
        lines = category.strip().split('\n')
        category_name = lines[0].strip('*')
        details = {}
        for line in lines[1:]:
            if line.startswith('* **Description:**'):
                details['description'] = line.split('**Description:**')[1].strip()
            elif line.startswith('* **Research Prompt:**'):
                details['research_prompt'] = line.split('**Research Prompt:**')[1].strip().strip('*')
            elif line.startswith('* **Key Concepts:**'):
                details['key_concepts'] = [concept.strip() for concept in line.split('**Key Concepts:**')[1].strip().split(',')]
        research_dict[category_name] = details
    return research_dict

def create_prompt(topic, depth, focus):
    prompt = f"""Conduct a comprehensive research on the topic: "{topic}".
    Depth of research: {depth}/5
    Focus areas: {focus}

    Please provide:
    1. A brief overview of the topic
    2. Key findings or main points (bulleted list)
    3. Important details or explanations for each main point
    4. Any relevant statistics or data
    5. Current trends or future outlook
    6. Potential applications or implications
    7. Challenges or limitations
    8. Conclusion or summary

    Be precise and concise in your explanations."""
    return prompt

def query_perplexity(prompt, api_key):
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2000,
        "temperature": 0.2,
        "return_citations": True
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error querying Perplexity AI: {str(e)}")
        return None

def format_output(response):
    if not response:
        return "No response received from Perplexity AI."
    
    content = response['choices'][0]['message']['content']
    citations = response.get('citations', [])
    formatted_output = f"""
Research Results:

{content}

Sources:
"""
    for citation in citations:
        formatted_output += f"- {citation['url']}\n"
    return formatted_output

def generate_research_prompts(research_dict, num_prompts=4):
    prompts = []
    categories = list(research_dict.keys())
    for _ in range(num_prompts):
        category = random.choice(categories)
        category_data = research_dict[category]
        if random.choice([True, False]):
            prompt = category_data['research_prompt']
        else:
            concepts = category_data['key_concepts']
            concept = random.choice(concepts)
            prompt = f"How can {concept.lower()} be applied in {category.lower()} to improve solar farm efficiency?"
        prompts.append(prompt)
    return prompts

# Create the chain
chain = LLMChain(llm=llm, prompt=RESEARCH_PROMPT)

# Function to run the chain
def generate_research_categories(research_field, research_topic):
    return chain.run(research_field=research_field, research_topic=research_topic)

# Streamlit app
st.set_page_config(page_title="Research Prompt Generator and Query System", page_icon="📚", layout="wide")

st.title("Research Prompt Generator and Query System")

# Sidebar for app mode selection
app_mode = st.sidebar.selectbox("Choose the app mode",
    ["Generate Research Prompts", "Direct Perplexity AI Query"])

if app_mode == "Generate Research Prompts":
    col1, col2 = st.columns(2)
    
    with col1:
        research_field = st.text_input("Enter the research field:", "Using Sustainable methods to provide maintenance in solar farms")
        research_topic = st.text_input("Enter the research topic:", "HOW CAN WE USE DRONES IN SYSTEMS?")
    
    with col2:
        num_prompts = st.slider("Number of prompts to generate:", 1, 10, 4)
        depth = st.slider("Depth of research:", 1, 5, 3)

    if st.button("Generate Research Prompts and Query"):
        with st.spinner("Generating research prompts..."):
            result = generate_research_categories(research_field, research_topic)
            res = parse_research_categories(result)
            research_prompts = generate_research_prompts(res, num_prompts)

        st.subheader("Generated Research Prompts:")
        for i, prompt in enumerate(research_prompts, 1):
            st.write(f"{i}. {prompt}")

        st.subheader("Research Results:")
        for i, prompt in enumerate(research_prompts, 1):
            with st.spinner(f"Researching prompt {i}..."):
                full_prompt = create_prompt(prompt, depth, research_field)
                response = query_perplexity(full_prompt, PERPLEXITY_API_KEY)
                formatted_result = format_output(response)
                
                with st.expander(f"Research Result {i}"):
                    st.markdown(formatted_result)

elif app_mode == "Direct Perplexity AI Query":
    st.subheader("Direct Perplexity AI Query")
    user_prompt = st.text_area("Enter your research prompt:", 
        "Explain the impact of artificial intelligence on renewable energy systems.")
    
    if st.button("Query Perplexity AI"):
        with st.spinner("Querying Perplexity AI..."):
            response = query_perplexity(user_prompt, PERPLEXITY_API_KEY)
            formatted_result = format_output(response)
            
        st.subheader("Research Result:")
        st.markdown(formatted_result)

# Add some information about the app
st.sidebar.title("About")
st.sidebar.info(
    "This app offers two modes:\n\n"
    "1. Generate Research Prompts: Creates prompts based on a given field and topic, "
    "then uses Perplexity AI to conduct research on these prompts.\n\n"
    "2. Direct Perplexity AI Query: Allows you to input your own research prompt "
    "and get results directly from Perplexity AI.\n\n"
    "Choose your mode in the dropdown above."
)

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: black;
        text-align: center;
    }
    </style>
    <div class="footer">
        <p>Developed with ❤️ by Your Name/Company | © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
