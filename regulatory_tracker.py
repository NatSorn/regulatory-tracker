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

from crewai_tools import ScrapeWebsiteTool, WebsiteSearchTool

scrape_tool = ScrapeWebsiteTool(
    website='https://www.centralbank.ie/news-media/press-releases',

  )

web_search_tool = WebsiteSearchTool(
    website='https://www.centralbank.ie/news-media/press-releases',

)

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

 # Web Scraper Agent
scraper_agent = Agent(
  role='Web Scraper',
          goal='Efficiently scrape and extract content from Central Bank of Ireland website',
          backstory="""You are an expert web scraper specialized in extracting
          information from financial regulatory websites.""",
          verbose=True,
          allow_delegation=False
      )

      # Content Analyzer Agent
analyzer_agent = Agent(
          role='Content Analyzer',
          goal='Analyze content for Anti-Money Laundering relevance and extract key information. Check if the link is not accessible, do not scrap the content by yourself, but delegate the work back to the Web Scraper Agent and tell the Web Scraper Agent to make sure to get the correct link',
          backstory="""You are a financial regulation expert specialized in
          Anti-Money Laundering.""",
          verbose=True,
          allow_delegation=True
      )

# Task 1: Scrape Website
scraping_task = Task(
          description="""
          1. Visit the Central Bank of Ireland news-media section
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

      # Task 2: Analyze Content
analysis_task = Task(
          description="""
          1. Check the link if it redirect to the correct relevant updates page
          2. If the link is not accessible, do not scrap the content by yourself, but delegate the work back to the Web Scraper Agent and tell the Web Scraper Agent to make sure to get the correct link
          2. Review the scraped content
          3. Identify any mentions or relevance to Anti-Money Laundering
          4. Extract key points and regulatory implications
          5. Categorize the importance of each update
          """,
          agent=analyzer_agent,
          tools=[scrape_validate_link_tool, web_search_validate_link_tool],
          expected_output="A list of Anti-Money Laundering-related findings with key points, regulatory implications, and categorization of importance.",
          output_json=News,
      )

crew = Crew(
            agents=[scraper_agent, analyzer_agent],
            tasks=[scraping_task,analysis_task],
            memory=True,
            verbose=True,
        )

result = crew.kickoff()

import json
import pandas as pd
from datetime import datetime

# Get the raw output from CrewOutput

print(f"Raw Output: {result.raw}")
if result.json_dict:
    print(f"JSON Output: {json.dumps(result.json_dict, indent=2)}")
if result.pydantic:
    print(f"Pydantic Output: {result.pydantic}")
print(f"Tasks Output: {result.tasks_output}")
print(f"Token Usage: {result.token_usage}")

# Use st.write instead of st.print
st.text(f"Tasks Output: {result.tasks_output}")