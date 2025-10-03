import streamlit as st
import pandas as pd
import re
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from crewai import LLM

llm = LLM(
    model="openai/gpt-4o-mini", # call model by provider/model_name
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.1,
    frequency_penalty=0.1,
    presence_penalty=0.1
)

st.set_page_config(page_title="Regulatory Tracker", layout="wide")

st.title("Regulatory Tracker")

input = st.text_input("Enter your regulatory topic you want to retrieve the news", type="default")
submit = st.button("Submit")


from crewai_tools import ScrapeWebsiteTool, WebsiteSearchTool, ScrapeElementFromWebsiteTool


scrape_tool = ScrapeWebsiteTool(
    website='https://www.centralbank.ie/search-results#',

  )

web_search_tool = WebsiteSearchTool(
    website='https://www.centralbank.ie/search-results#',

)

# url_scrape_tool = ScrapeElementFromWebsiteTool()

scrape_validate_link_tool = ScrapeWebsiteTool()
web_search_validate_link_tool = WebsiteSearchTool()

import json

from crewai import Agent, Crew, Process, Task
from pydantic import BaseModel


# Define the Pydantic model for the blog
class News(BaseModel):
    Publisher_Name: str
    News_Title: str
    News_Summary: str
    News_Date: str
    News_Link: str

# Web Searcher Agent (move outside the News model)
searcher_agent = Agent(
    role='Web Searcher',
    goal='Efficiently search and identify relevant {input} from Central Bank of Ireland website',
    backstory="""You are an expert web searcher specialized in finding
    information from financial regulatory websites.""",
    verbose=True,
    allow_delegation=False
)

scraper_agent = Agent(
    role='Web Scraper',
    goal='Efficiently scrape and extract search result under News & Media category from Central Bank of Ireland website',
    backstory="""You are an expert web scraper specialized in extracting
    information from financial regulatory websites.""",
    verbose=True,
    allow_delegation=False
)

# url_scraper_agent = Agent(
#     role='URL Scraper',
#     goal='Scrape the full URL link of each news url from span.path element by using the Scrape Element From Website Tool',
#     backstory="""You are an expert web scraper specialized in extracting
#     information from financial regulatory websites.""",
#     verbose=True,
#     allow_delegation=False
# )

# Content Analyzer Agent
analyzer_agent = Agent(
    role='Content Analyzer',
    goal='Analyze content for {input} relevance and extract key information. Check if the link is not accessible, do not scrap the content by yourself, but delegate the work back to the Web Scraper Agent and tell the Web Scraper Agent to make sure to get the correct link',
    backstory="""You are a financial regulation expert specialized in
    {input}.""",
    verbose=True,
    allow_delegation=True
)

searching_task = Task(
          description="""
          1. Visit the https://www.centralbank.ie/search-results#
          2. Put {input} in the search bar to search for relevant news
          """,
          agent=searcher_agent,
          tools=[scrape_tool, web_search_tool],
          expected_output="A list of search results"
      )


scraping_task = Task(
          description="""
          1. Visit the search result under News & Media category 
          2. Identify all updates from March 1, 2025 to present
          3. Extract all content including dates, titles
          4. Generate summary of the news
          4. Extract the link of the specific news in this format, for example1: https://www.centralbank.ie/news/article/the-central-bank-takes-enforcement-action-against-swilly-mulroy-credit-union-for-breaches-of-anti-money-laundering-requirements , for example2: https://www.centralbank.ie/news/article/press-release-derville-rowland-appointed-to-executive-board-of-new-eu-authority-for-anti-money-laundering-23-May-25
          5. Return the data in a structured format
          """,
          agent=scraper_agent,
          tools=[scrape_tool, web_search_tool],
          expected_output="A structured dataset containing dates, titles, full text, and links of each relevant updates.",
          output_json=News,
      )

# url_scraping_task = Task(
#           description="""
#           1. Scrape the url in the span.path element of each news from the search result by using the Scrape Element From Website Tool
#           2. Return the data in a News_Link field in the structured format
#           """,
#           agent=url_scraper_agent,
#           tools=[url_scrape_tool],
#           expected_output="A list of full URL links for each news article.",
#           output_json=News,
#       )

      # Task 2: Analyze Content
analysis_task = Task(
          description="""
          1. Check the link if it redirect to the correct relevant updates page
          2. If the link is not accessible, do not scrap the content by yourself, but delegate the work back to the Web Scraper Agent and tell the Web Scraper Agent to make sure to get the correct link
          2. Review the scraped content
          3. Identify any mentions or relevance to {input}
          4. Extract key points and regulatory implications
          5. Categorize the importance of each update
          """,
          agent=analyzer_agent,
          tools=[scrape_validate_link_tool, web_search_validate_link_tool],
          expected_output="A list of {input}-related findings with key points, regulatory implications, and categorization of importance.",
          output_json=News,
      )

crew = Crew(
            agents=[searcher_agent, scraper_agent, analyzer_agent],
            tasks=[searching_task, scraping_task,analysis_task],
            memory=True,
            verbose=True,
        )

import json
import pandas as pd
from datetime import datetime
import io
import streamlit as st
import re

if submit and input:

    with st.spinner('AI Agent processing...'):
        result = crew.kickoff()

        print(f"Raw Output: {result.raw}")
        df = None
        try:
            if isinstance(result.raw, list):
                df = pd.DataFrame(result.raw)
            elif isinstance(result.raw, dict):
                df = pd.DataFrame([result.raw])
            elif isinstance(result.raw, str):
                # Try to extract all JSON arrays or objects from the string
                matches = re.findall(r'(\{.*?\}|\[.*?\])', result.raw, re.DOTALL)
                for match in matches:
                    try:
                        raw_json = json.loads(match)
                        if isinstance(raw_json, list):
                            df = pd.DataFrame(raw_json)
                            break
                        elif isinstance(raw_json, dict):
                            df = pd.DataFrame([raw_json])
                            break
                    except Exception:
                        continue
                if df is None:
                    raise ValueError("No valid JSON found in Raw Output.")
        except Exception as e:
            st.warning(f"Could not convert Raw Output to DataFrame: {e}")

        if df is not None and not df.empty:
            col1, col2 = st.columns([4, 1])
            with col2:
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Export to CSV",
                    data=csv_buffer.getvalue(),
                    file_name="regulatory_news.csv",
                    mime="text/csv"
                )
            st.table(df)
        else:
            st.warning("No tabular data found in Raw Output.")
        if result.pydantic:
            print(f"Pydantic Output: {result.pydantic}")
        print(f"Tasks Output: {result.tasks_output}")
        print(f"Token Usage: {result.token_usage}")