import streamlit as st
from streamlit_multipage import MultiPage
from streamlit_option_menu import option_menu
from streamlit_echarts import st_echarts
import pandas as pd
import boto3
import botocore
import os
import anthropic
import yfinance as yf
from datetime import datetime, timedelta
import json
import numpy as np
import re

# Set AWS credentials and region
os.environ["AWS_ACCESS_KEY_ID"] = "AWS"
os.environ["AWS_SECRET_ACCESS_KEY"] = "AWS"
os.environ["REGION_NAME"] = "AWS"

# Setup AWS session for S3
boto3.setup_default_session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                            region_name=os.getenv('REGION_NAME'))

ANTHROPIC_API_KEY = "API"
CLAUDE_HAIKU = "claude-3-haiku-20240307"
CLAUDE_SONNET = "claude-3-sonnet-20240229"

class AWSOperations:
    def __init__(self):
        self.s3 = boto3.client('s3')

    def fetch_object(self, file_name, bucket_name):
        obj = self.s3.get_object(Bucket=bucket_name, Key=file_name)
        return obj['Body'].read().decode('utf-8')

class AIResponseGenerator:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_response(self, prompt, system_prompt, partner_letters, fund_names_dates):
        client = anthropic.Anthropic(api_key=self.api_key)
        
        # Create XML tags for each document
        tagged_letters = []
        for letter, fund_name_date in zip(partner_letters, fund_names_dates):
            fund_name, date = fund_name_date.lower().split(" ", 1)
            fund_name = fund_name.replace(" ", "")
            quarter = date.split(" ")[0].lower()
            year = date.split(" ")[1]
            tag = f"<{fund_name}_{year}_{quarter}>"
            tagged_letter = f"{tag}\n{letter}\n</{fund_name}_{year}_{quarter}>"
            tagged_letters.append(tagged_letter)
        
        combined_letters = "\n\n".join(tagged_letters)
        
        message = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=2000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{combined_letters}\n\n{prompt}"
                        }
                    ]
                }
            ]
        )
        
        raw_text = message.content
        answer = raw_text[0].text
        
        # Extract the content within <answer></answer> tags if present
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        st.write(answer)

class DocumentFetcher:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_partner_letters(self, fund_names_dates):
        partner_letters = []
        for fund_name_date in fund_names_dates:
            parts = fund_name_date.split()
            fund_name = " ".join(parts[:-2]).lower().replace(" ", "")
            date = " ".join(parts[-2:])
            file_name = f"{fund_name}/cleaned/{fund_name_date}.txt"
            try:
                letter_content = self.aws_operations.fetch_object(file_name, "hedgefunds")
                partner_letters.append(letter_content)
            except Exception as e:
                print(f"File not found: {file_name}")
                print(f"Error: {str(e)}")
        return partner_letters

def fetch_fund_names(aws_operations, bucket_name, fund_info_path):
    # Fetch the JSON file from S3
    json_data = aws_operations.fetch_object(fund_info_path, bucket_name)
    fund_info_data = json.loads(json_data)

    # Get unique fund names
    fund_names = set(obj['Fund Name'].replace(", LP", "") for obj in fund_info_data)

    return list(fund_names)

# def select_funds(aws_operations, bucket_name, fund_info_path):
#     fund_names = fetch_fund_names(aws_operations, bucket_name, fund_info_path)
#     selected_funds = st.sidebar.multiselect("Select Funds", fund_names)
#     return selected_funds

class OpportunityScout:
    def __init__(self, aws_operations, bucket_name):
        self.aws_operations = aws_operations
        self.bucket_name = bucket_name

    def fetch_json_data(self, formatted_fund_name):
        json_file_path = f"{formatted_fund_name}/{formatted_fund_name}_equities.json"

        try:
            json_data = self.aws_operations.fetch_object(json_file_path, self.bucket_name)
            json_data = json.loads(json_data)
            return json_data
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"JSON file not found for fund: {formatted_fund_name}")
                return None
            else:
                raise e

    def filter_companies(self, json_data, sectors, start_quarter, end_quarter, pitched, exited):
        filtered_companies = []

        for company in json_data:

            if sectors and company['Sector'] not in sectors:
                continue

            pitch_quarter = company['Date']

            if start_quarter and end_quarter:
                if pitch_quarter < start_quarter or pitch_quarter > end_quarter:
                    continue

            if pitched and str(company['PositionOpen']) in ['1', '0']:
                if str(company['PositionOpen']) == '0':
                    continue
            if exited and str(company['PositionClose']) in ['1', '0']:
                if str(company['PositionClose']) == '0':
                    continue

            filtered_companies.append(company)

        return filtered_companies
    
    def aggregate_companies(self, selected_funds, sectors, start_quarter, end_quarter, pitched, exited):
        aggregated_companies = []

        for fund_name in selected_funds:
            json_data = self.fetch_json_data(fund_name)
            filtered_companies = self.filter_companies(json_data, sectors, start_quarter, end_quarter, pitched, exited)
            aggregated_companies.extend(filtered_companies)

        return aggregated_companies

    def display_companies(self, aggregated_companies):
        if not aggregated_companies:
            st.write("No companies found matching the selected criteria.")
            return

        df = pd.DataFrame(aggregated_companies)
        
        # Remove the "Description" column from the displayed dataframe
        if "Description" in df.columns:
            df = df.drop(columns=["Description"])
        
        # Add the message to inform the user about double-clicking on a cell
        st.write("**Double-click on a cell to see the full text.**")
        
        st.dataframe(df)

    def get_top_sectors(self, selected_funds, start_quarter, end_quarter):
        sector_counts = {}

        for fund_name in selected_funds:
            json_data = self.fetch_json_data(fund_name)
            filtered_companies = self.filter_companies(json_data, None, start_quarter, end_quarter, None, None)

            for company in filtered_companies:
                sector = company['Sector']
                if sector in sector_counts:
                    sector_counts[sector] += 1
                else:
                    sector_counts[sector] = 1

        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        top_sectors = sorted_sectors[:3]

        return top_sectors

    def run(self, fund_type, selected_funds):
        if not selected_funds:
            st.write("**Please select at least one fund in the side bar**")
            return

        quarters = ["2022 Q3", "2022 Q4", "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1"]
        start_quarter, end_quarter = st.sidebar.select_slider("Select Date Range", options=quarters, value=(quarters[0], quarters[-1]))

        if selected_funds:
            top_sectors = self.get_top_sectors(selected_funds, start_quarter, end_quarter)
            if top_sectors:
                st.write(f"**Top 3 most discussed sectors by the selected funds from {start_quarter} to {end_quarter}:**")
                for sector, count in top_sectors:
                    st.write(f"- {sector} ({count} mentions)")
            else:
                st.write(f"No sectors discussed by the selected funds from {start_quarter} to {end_quarter}.")

        sectors = ["All", "Financials", "Energy", "Health Care", "Communication Services", "Industrials", "Information Technology", "Consumer Discretionary", "Real Estate"]
        st.write("**Add some filters below and extract the investment opportunities discussed in the partner letters:**")
        selected_sectors = st.multiselect("Select Sectors", sectors)

        # Handle the "All" option
        if "All" in selected_sectors:
            selected_sectors = sectors[1:]  # Include all sectors except "All"

        pitched = st.checkbox("Newly Added Positions")
        exited = st.checkbox("Exited Positions")

        if st.button("Submit"):
            aggregated_companies = self.aggregate_companies(selected_funds, selected_sectors, start_quarter, end_quarter, pitched, exited)
            self.display_companies(aggregated_companies)
                 
class PerformancePulse:
    def __init__(self, aws_operations, ai_response_generator, document_fetcher):
        self.aws_operations = aws_operations
        self.ai_response_generator = ai_response_generator
        self.document_fetcher = document_fetcher

    def fetch_performance_data(self, selected_funds, start_quarter, end_quarter):
        # Fetch the performance data from the JSON file in the S3 bucket
        performance_data = self.aws_operations.fetch_object("hedgefund_performance_insights.json", "hedgefunds")
        performance_data = json.loads(performance_data)

        # Filter the performance data based on the selected funds and date range
        filtered_data = [
            obj for obj in performance_data
            if obj['Fund Name'] in selected_funds and start_quarter <= obj['Date'] <= end_quarter
        ]

        return filtered_data

    def display_performance_table(self, filtered_data):
        # Extract the relevant columns from the filtered data for the performance table
        table_data = [
            {
                'Fund Name': obj['Fund Name'],
                'Date': obj['Date'],
                'Quarterly Performance Net of Fees': self.convert_to_percentage(obj.get('Quarterly Performance Net of Fees', ''))
            }
            for obj in filtered_data
        ]

        # Create a DataFrame from the table data and sort it by 'Quarterly Performance Net of Fees' in descending order
        df = pd.DataFrame(table_data)
        df = df.sort_values('Quarterly Performance Net of Fees', ascending=False, key=lambda x: pd.to_numeric(x.str.rstrip('%'), errors='coerce'))

        # Limit the table to the top 5 best performing quarters
        top_5_df = df.head(5)

        st.table(top_5_df)

    def convert_to_percentage(self, value):
        # Convert a value to percentage format
        try:
            return f"{float(value):.1f}%"
        except (ValueError, TypeError):
            return value
        
    def convert_to_float(self, value):
        # Convert a value to float if possible, otherwise return the original value
        try:
            return float(value)
        except (ValueError, TypeError):
            return value

    def handle_dropdown_selection(self, selected_option, filtered_data):
        # Extract the relevant insights based on the selected option
        if selected_option in ['Key Contributors to Performance', 'Key Detractors from Performance']:
            insight_key = selected_option
            message_prompt = f"""
            Provide a high-level overview of the main {insight_key} across the selected hedge funds and quarters. 
            Structure your response in the following format:

            1. Start with the most recent quarter and work backwards chronologically.
            2. For each quarter:
                a. Point: Make a clear and concise point about the main {insight_key} for that quarter.
                b. Evidence: Provide specific examples from the extracted insights to support your point. Include the fund name for each example.
                c. Explain: Dive deeper into the reasons behind the {insight_key}. Explain why certain stocks or sectors performed well or poorly. Also, if more than one fund was selected, provide a summary that compares and contrasts the different funds and identifies any common themes or divergences across the entire time period.
            
            IMPORTANT RULES:
            - Each quarter should be analyzed in a separate paragraph.
            - Do not include the words "Point" or "Evidence" in your final response. Instead, provide a clear and concise analysis.
            - Include an overview of each fund that appears in the aggregated insights for each quarter. Do not skip any funds.
            - Use the format [Fund Name, Quarter] when citing specific examples.
            """
            system_prompt = f"""
            You are an experienced investment analyst reviewing the quarterly partner letters from various hedge funds. I have extracted the '{insight_key}' section from each letter and aggregated them for you. 
            Your task is to provide a high-level overview of the main {insight_key} while also diving into the details. Make clear points about the overall trends or patterns you observe. Then, support your point with specific examples from the extracted insights, including the fund name and quarter for each example. 
            Finally, explain the reasons behind the {insight_key}. Analyze why certain stocks or sectors performed the way they did. Throughout your response, compare and contrast the different funds and highlight any common themes or notable differences.
            After reviewing the relevant information, take a moment to organize your thoughts within <thinking></thinking> tags before presenting your final analysis to the user within <answer></answer> tags.
            """
        elif selected_option == 'Portfolio Positioning and Adjustments':
            insight_key = selected_option
            message_prompt = """
            Please provide a summary of the key portfolio positioning changes and adjustments made by the selected hedge funds over the specified time period.
            Structure your response in the following format:

            1. Start with the most recent quarter and work backwards chronologically.
            2. For each quarter:
                a. Point: Make a clear and concise point about the main portfolio positioning changes and adjustments for that quarter. Cite which fund letters you are referring to.
                b. Evidence: Provide specific examples from the extracted insights to support your point. Include the fund name and specific details about the equity positions, backed up with quantitative data if possible.
                c. Explain: Analyze how these positioning changes aligned with or diverged from the funds' overall investment philosophies and market conditions. Assess the effectiveness of these adjustments based on the subsequent performance data. Also, if more than one fund was selected, provide a summary that compares and contrasts the different funds and identifies any common themes or divergences across the entire time period.

            Provide a comprehensive paragraph for each point. Do not return a list of bullet points; instead, write a well-crafted paragraph for each point.
            IMPORTANT RULES:
            - Each quarter should be analyzed in a separate paragraph.
            - Do not include the words "Point" or "Evidence" in your final response. Instead, provide a clear and concise analysis.
            - Include an overview of each fund that appears in the aggregated insights for each quarter. Do not skip any funds.
            - Use the format [Fund Name, Quarter] when citing specific examples.
            """
            system_prompt = """
            As an experienced investment analyst, you have been tasked with reviewing the 'Portfolio Positioning and Adjustments' sections extracted from the quarterly partner letters of various hedge funds.
            Your goal is to provide a comprehensive overview of the key changes in portfolio positioning and the rationale behind these adjustments.
            First, make a clear point about the main trends or patterns you observe in the portfolio positioning changes. Then, support your point with specific examples from the extracted insights, including the fund name and quarter for each example.
            Next, analyze how these positioning changes aligned with or diverged from the funds' overall investment philosophies and market conditions. Assess the effectiveness of these adjustments based on the subsequent performance data.
            Identify any significant shifts in strategy, sector exposure, or individual positions.
            Throughout your response, highlight common themes and notable contrasts among the funds.
            After collecting your thoughts, outline your response within <thinking></thinking> tags before presenting your final analysis to the user within <answer></answer> tags.
            """
        
        # Extract the relevant insights data from the filtered data
        insight_data = [
            {
                'Fund Name': obj['Fund Name'],
                'Date': obj['Date'],
                'Insight': obj.get(insight_key, '')
            }
            for obj in filtered_data
        ]

        # Format the insights data for sending to the AI API
        formatted_insights = []
        for obj in insight_data:
            formatted_insight = f"[{obj['Fund Name']}, {obj['Date']}]\n{obj['Insight']}"
            formatted_insights.append(formatted_insight)

        aggregated_insights = "\n\n".join(formatted_insights)

        return message_prompt, system_prompt, aggregated_insights

    def generate_performance_pulse_response(self, prompt, system_prompt, aggregated_insights):
        client = anthropic.Anthropic(api_key=self.ai_response_generator.api_key)

        message = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=3000,
            temperature=0.3,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{aggregated_insights}\n\n{prompt}"
                        }
                    ]
                }
            ]
        )

        raw_text = message.content
        answer = raw_text[0].text

        # Extract the content within <answer></answer> tags if present
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        st.write(answer)

    def fetch_positioning_text(self, fund_name, quarter):
        # Fetch the performance data from the JSON file in the S3 bucket
        performance_data = self.aws_operations.fetch_object("hedgefund_performance_insights.json", "hedgefunds")
        performance_data = json.loads(performance_data)
        
        # Find the matching fund and quarter in the performance data
        for obj in performance_data:
            if obj['Fund Name'] == fund_name and obj['Date'] == quarter:
                return obj.get('Portfolio Positioning and Adjustments', '')
        
        return ''
    
    def run(self, selected_funds):
        quarters = ["2022 Q3", "2022 Q4", "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1"]
        start_quarter, end_quarter = st.sidebar.select_slider("Select Date Range", options=quarters, value=(quarters[0], quarters[-1]))

        if selected_funds:
            filtered_data = self.fetch_performance_data(selected_funds, start_quarter, end_quarter)
            self.display_performance_table(filtered_data)
            
            # Filter out rows with invalid numeric values in 'Quarterly Performance Net of Fees'
            valid_data = [row for row in filtered_data if isinstance(self.convert_to_float(row.get('Quarterly Performance Net of Fees', '')), float)]
            
            if valid_data:
                # Get the best performing fund and quarter from the valid data
                best_performing_row = max(valid_data, key=lambda x: self.convert_to_float(x.get('Quarterly Performance Net of Fees', '')))
                best_performing_fund = best_performing_row['Fund Name']
                best_performing_quarter = best_performing_row['Date']
                
                # Fetch the "Portfolio Positioning and Adjustments" text for the best performing fund and quarter
                positioning_text = self.fetch_positioning_text(best_performing_fund, best_performing_quarter)
                
                if positioning_text:
                    st.subheader(f"{best_performing_fund} was the best performing fund given the chosen filters. This was their fund positioning for {best_performing_quarter}:")
                    st.write(positioning_text)
                    st.write("<font color='blue'>**Please limit the date range in the side bar to 3 quarters!**</font>")
            else:
                st.write("No valid performance data available for the selected funds and date range.")

            selected_option = st.selectbox("Select an option", ["Key Contributors to Performance", "Key Detractors from Performance", "Portfolio Positioning and Adjustments"])

            if selected_option:
                message_prompt, system_prompt, aggregated_insights = self.handle_dropdown_selection(selected_option, filtered_data)
                    
                if st.button("Submit"):
                    self.generate_performance_pulse_response(message_prompt, system_prompt, aggregated_insights)
        else:
            st.write("**Please select at least one fund to view performance data.**")

class MarketMoodMonitor:
    def __init__(self, aws_operations, ai_response_generator, fund_info_path, document_fetcher):
        self.aws_operations = aws_operations
        self.ai_response_generator = ai_response_generator
        self.fund_info_path = fund_info_path
        self.document_fetcher = document_fetcher

    def fetch_fund_info_data(self):
        fund_info_data = self.aws_operations.fetch_object(self.fund_info_path, "hedgefunds")
        fund_info_data = json.loads(fund_info_data)
        return fund_info_data
    
    def get_unique_values(self, fund_info_data, key):
        values = set()
        for obj in fund_info_data:
            if key in obj:
                values.update(obj[key].split(", "))
        return list(values)

    def handle_theme_specific(self, fund_info_data, analysis_type, selected_funds, start_quarter, end_quarter):
        if analysis_type == 'Market Commentary':
            themes = self.get_unique_values(fund_info_data, 'Macro')
        elif analysis_type == 'Asset Class':
            themes = self.get_unique_values(fund_info_data, 'Asset Classes')
        elif analysis_type == 'Geography':
            themes = self.get_unique_values(fund_info_data, 'Geographies')

        selected_themes = st.multiselect(f'Select {analysis_type.lower()} themes:', themes)

        if selected_themes:
            # Add text to inform users about uploading documents
            st.write("**You can upload a maximum of 5 documents to filter in the sidebar.**")

            # Filter the fund_info_data based on the selected themes
            filtered_funds_data = []
            for obj in fund_info_data:
                if analysis_type == 'Market Commentary' and any(theme in obj.get('Macro', '') for theme in selected_themes):
                    filtered_funds_data.append(obj)
                elif analysis_type == 'Asset Class' and any(theme in obj.get('Asset Classes', '') for theme in selected_themes):
                    filtered_funds_data.append(obj)
                elif analysis_type == 'Geography' and any(theme in obj.get('Geographies', '') for theme in selected_themes):
                    filtered_funds_data.append(obj)

            # Filter the filtered_funds_data based on the selected date range
            filtered_funds_data = [obj for obj in filtered_funds_data if start_quarter <= obj['Date'] <= end_quarter]

            # Filter the filtered_funds_data based on the selected funds
            if selected_funds:
                filtered_funds_data = [obj for obj in filtered_funds_data if obj['Fund Name'] in selected_funds]

            if filtered_funds_data:
                # Get the fund names and dates from the filtered data
                fund_names_dates = [f"{obj['Fund Name']} {obj['Date']}" for obj in filtered_funds_data]

                st.write(f"Funds and quarters that discussed the selected {analysis_type.lower()} themes:")
                for fund_name_date in fund_names_dates:
                    st.write(f"- {fund_name_date}")

                if st.button('Submit'):
                    message_prompt = f"Provide a detailed overview of the {', '.join(selected_themes)} themes discussed in the selected partner letters. Break it down into clear, structured bullet points. Highlight the key points that the letters discussed. Please compare and contrast between the different funds and quarters. Make sure to cite your sources by putting the title of the letter that was cited in brackets like this: [Greenlight Capital 2023 Q4]. I want to know the exactly source of each view point. The purpose is to make the user aware of the outlook on these specific themes. Please present your findings in a well-structured, easy-to-follow format."
                    
                    system_prompt = f"You are an experienced investment analyst with a deep understanding of various {analysis_type.lower()} themes. I have attached partner letters from the selected hedge funds for you to analyze and reference for the upcoming task. Each letter is identified by the appropiate XML tags at the top and bottom of the letter. Each hedge fund writes a quarterly partner letter discussing topics such as their performance, \
                        macroeconomic views, and rationale for adding specific equity positions to their fund. Please carefully read through the entire document and identify the most relevant commentary related to the selected themes: {', '.join(selected_themes)}. When you complete your task, first plan how you should answer and which data you will use within \
                            <thinking> </thinking> XML tags. This is a space for you to write down relevant content and will not be shown to the user. Once you are done thinking, output your final answer to the user within <answer> </answer> XML tags. Do not include closing tags or unnecessary open-and-close tag sections."

                    partner_letters = self.document_fetcher.fetch_partner_letters(fund_names_dates)
                    
                    # Display the included document names
                    st.write("These funds were included in the analysis:")
                    for fund_name_date in fund_names_dates:
                        st.write(f"- {fund_name_date}")

                    # Generate the response
                    self.ai_response_generator.generate_response(message_prompt, system_prompt, partner_letters, fund_names_dates)
            else:
                st.write(f"No funds found that discussed the selected {analysis_type.lower()} themes.")

    def run(self, selected_funds):
        analysis_type = st.radio('Select analysis type:', ['Market Commentary', 'Asset Class', 'Geography'])
        fund_info_data = self.fetch_fund_info_data()
            
        # Add a slider for selecting the date range
        quarters = ["2022 Q3", "2022 Q4", "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1"]
        start_quarter, end_quarter = st.sidebar.select_slider("Select Date Range", options=quarters, value=(quarters[0], quarters[-1]))

        self.handle_theme_specific(fund_info_data, analysis_type, selected_funds, start_quarter, end_quarter)


class SpecificFundsSection:
    def __init__(self, aws_operations, ai_response_generator, document_fetcher):
        self.aws_operations = aws_operations
        self.ai_response_generator = ai_response_generator
        self.document_fetcher = document_fetcher

    def fetch_performance_data(self, selected_fund, start_quarter, end_quarter):
        # Fetch the performance data from the JSON file in the S3 bucket
        performance_data = self.aws_operations.fetch_object("hedgefund_performance_insights.json", "hedgefunds")
        performance_data = json.loads(performance_data)

        # Filter the performance data based on the selected fund and date range
        filtered_data = [
            obj for obj in performance_data
            if obj['Fund Name'] == selected_fund and start_quarter <= obj['Date'] <= end_quarter
        ]

        return filtered_data
    
    def fetch_available_dates(self, selected_fund):
        # Fetch the JSON file containing the fund information
        fund_info_data = self.aws_operations.fetch_object("hedgefund_general_insights.json", "hedgefunds")
        fund_info_data = json.loads(fund_info_data)

        # Extract the available dates for the selected fund
        available_dates = [obj['Date'] for obj in fund_info_data if obj['Fund Name'] == selected_fund]

        return sorted(set(available_dates))

    def display_line_graph(self, filtered_data):
        x_data = []
        y_data = []

        for obj in filtered_data:
            x_data.append(obj['Date'])
            performance = obj.get('Quarterly Performance Net of Fees', '')
            try:
                y_data.append(float(performance))
            except (ValueError, TypeError):
                y_data.append(None)

        option = {
            "xAxis": {
                "type": "category",
                "data": x_data,
            },
            "yAxis": {"type": "value"},
            "series": [{"data": y_data, "type": "line"}],
        }

        st_echarts(options=option, height="400px")

    def handle_performance_button_click(self, selected_fund, quarters_in_range, selected_performance_option):
        fund_names_dates = [f"{selected_fund} {quarter}" for quarter in quarters_in_range]
        partner_letters = self.document_fetcher.fetch_partner_letters(fund_names_dates)

        if selected_performance_option == 'Key Contributors to Performance':
            message_prompt = "Analyze the key positive contributors to {selected_fund}'s performance across the provided quarterly letters. Identify the top stocks, sectors, strategies or positions that drove outperformance each quarter. For each key contributor, extract specific evidence and examples from all relevant letters, clearly citing the quarter. Compare and contrast how that contributor performed across different quarters - highlight quarters where it was a top performer as well as any periods of underperformance if that exists. Explain the reasons and market conditions behind the diverging performance based on context from the letters."
            system_prompt = f"""
            You are reviewing the quarterly letters from {selected_fund} to analyze the top 3 positive performance contributors. If you cannot identify at least 5 companies, then only return the ones that you can find information on. For each of the top 3 contributors, provide a brief analysis structured as follows:
            Top Contributor 1:
            [1-2 sentence summary of the contributor, e.g. Stock ABC, Technology Sector, etc.]
            Key Quarters of Outperformance:
            [Cite quarter/period and supporting evidence from letter(s) showing strong performance]
            [Cite another quarter/period and evidence, if applicable]

            Rationale:
            [1-2 sentence explanation for why this contributor outperformed, based on context from letters]
            
            Top Contributor 2:
            ...
            IMPORTANT RULES:
            - Prioritize the most recent information if there are discrepancies across letters.
            - Avoid speculation beyond what is stated in the letters.
            - Note any limitations in your analysis due to lack of context or evidence.
            
            The key is to provide a holistic perspective on each major contributor by contrasting its performance across all available quarters/periods using evidence from the letters. This context is critical for accurately assessing contribution drivers.
            After collecting your thoughts, outline your response within <thinking></thinking> tags before presenting your final analysis to the user within <answer></answer> tags.
            """
        elif selected_performance_option == 'Key Detractors from Performance':
            message_prompt = "Analyze the key detractors from performance to {selected_fund}'s performance across the provided quarterly letters. Identify the key detracting stocks, sectors, strategies or positions that drove outperformance each quarter. For each key detractor, extract specific evidence and examples from all relevant letters, clearly citing the quarter. Compare and contrast how that contributor performed across different quarters - highlight quarters where it was a bad performer as well as any periods of top performance if that exists. Explain the reasons and market conditions behind the diverging performance based on context from the letters."
            system_prompt = f"""
            You are reviewing the quarterly letters from {selected_fund} to analyze the top 3 detractors. If you cannot identify at least 3 companies, then only return the ones that you can find information on. For each of the top 3 contributors, provide a brief analysis structured as follows:
            Top Detractor 1:
            [1-2 sentence summary of the detractor, e.g. Stock ABC, Technology Sector, etc.]
            Key Quarters of Underperformance:
            [Cite quarter/period and supporting evidence from letter(s) showing weak performance]
            [Cite another quarter/period and evidence, if applicable]
            Rationale:
            [1-2 sentence explanation for why this contributor outperformed, based on context from letters]
            
            Top Detractor 2:
            ...

            IMPORTANT RULES:
            - Prioritize the most recent information if there are discrepancies across letters.
            - Avoid speculation beyond what is stated in the letters.
            - Note any limitations in your analysis due to lack of context or evidence.
            
            The key is to provide a holistic perspective on each major detractor by contrasting its performance across all available quarters/periods using evidence from the letters. This context is critical for accurately assessing contribution drivers.
            After collecting your thoughts, outline your response within <thinking></thinking> tags before presenting your final analysis to the user within <answer></answer> tags.
            """
        elif selected_performance_option == 'Portfolio Positioning and Adjustments':
            message_prompt = "Provide a summary of the key portfolio positioning changes and adjustments made by {selected_fund} across the provided quarterly letters. Highlight any significant shifts in strategy, sector exposure, market cap focus, themes/investment theses, or notable individual position adjustments. For each high-level summary point, back it up with specific evidence extracted from the letters, clearly citing the quarter/timeframe. Analyze the rationale and effectiveness of major adjustments if mentioned in the letters or evident from performance data."
            system_prompt = f"""
            You are an experienced investment analyst. You are reviewing the portfolio positioning details across the quarterly letters from the hedge fund {selected_fund}. Your goal is to concisely summarize the key adjustments made to the fund's positioning, backing up each point with specific evidence from the letters.
            Please follow these steps:

            Review the letters chronologically, identifying any changes mentioned in positioning across asset classes, strategies, sectors, market caps, themes or individual positions compared to prior letters.
            Distill these changes into a few high-level summary points covering the key shifts in positioning focus over the period.
            For each summary point, immediately provide the supporting evidence extracted from the letters, clearly citing the timeframe/quarter for each piece of evidence.
            For major positioning adjustments, analyze the rationale based on the context provided in the letters. If available, evaluate the effectiveness based on any stated or observable performance impact.
            Throughout, highlight nuanced aspects like shifts towards small-caps, large-caps, fixed income, value, growth, shorting, thematic bets on areas like AI/technology, deep value, spin-offs etc.

            Please structure your response as follows:
            High-level Positioning Summary Point 1:
            [Summary statement]
            Supporting Evidence:

            [Details from letter 1, quarter/timeframe cited]
            [Details from letter 2, quarter/timeframe cited]
            Rationale/Impact Analysis:
            [Based on context in letters]

            High-level Positioning Summary Point 2:
            [Summary statement]
            Supporting Evidence:
            ...
            Limitations:
            [Any caveats about missing context or inability to fully evaluate adjustments]
            """

        return message_prompt, system_prompt, partner_letters, fund_names_dates
    
    def fetch_json_data(self, selected_fund):
        # Format the fund name for the JSON file path
        formatted_fund_name = selected_fund.lower().replace(" ", "")
        json_file_path = f"{formatted_fund_name}/{formatted_fund_name}_equities.json"

        try:
            # Fetch the JSON data from S3
            json_data = self.aws_operations.fetch_object(json_file_path, "hedgefunds")
            json_data = json.loads(json_data)
            return json_data
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                print(f"JSON file not found for fund: {selected_fund}")
                return None
            else:
                raise e
            
    def get_top_sectors(self, selected_fund, start_quarter, end_quarter):
        # Fetch the JSON data for the selected fund
        json_data = self.fetch_json_data(selected_fund)

        if json_data:
            # Filter the data based on the selected date range
            filtered_data = [item for item in json_data if start_quarter <= item['Date'] <= end_quarter]

            # Count the occurrences of each sector
            sector_counts = {}
            for item in filtered_data:
                sector = item['Sector']
                if sector in sector_counts:
                    sector_counts[sector] += 1
                else:
                    sector_counts[sector] = 1

            # Sort the sectors by count in descending order
            sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)

            # Return the top 3 sectors
            return sorted_sectors[:3]
        else:
            return []
        
    def run(self, selected_fund):
        if selected_fund:
            st.title(selected_fund)
            st.subheader(f"Investment Type: Hedge Funds")
            
            if selected_fund == "Greenlight Capital":
                st.write("""
                Greenlight Capital is based in New York City, New York.\n
                Investment Horizon: 1-5 years\n
                Portfolio Size: 20 to 30 securities\n
                AUM: $1.4 billion\n
                Greenlight Capital is a value-oriented hedge fund founded by David Einhorn in 1996. The fund employs a long/short investment strategy. 
                """)
            elif selected_fund == "Maran Capital Management":
                st.write("""
                Maran Capital Management is based in Denver, Colorado.\n
                Investment Horizon: 3-5 years\n
                Portfolio Size: 10 securities\n
                Maran Capital Management is commitment to disciplined value investing aiming to find securities trading at 50 percent or less of intrinsic value.
                """)
            
            elif selected_fund == "OKeefe Stevens":
                st.write("""
                OKeefe Stevens Advisory is based in Rochester, New York.\n
                Investment Horizon: 3-5 years\n
                AUM: $0.2 billion\n
                OKeefe Stevens is a value-oriented is an investment management firm. 
                """)
            
            elif selected_fund == "White Brook Capital":
                st.write("""
                White Brook Capital is based in Chicago, Illinois.\n
                Investment Horizon: 3-5 years\n
                Portfolio Size: 10 securities\n
                AUM: $20 million\n
                White Brooke Capital focuses on mid-cap and considers the ESG scores when researching equities. The fund prioritizes bottom-up fundamental research over macroeconomic considerations. 
                """)
            
            elif selected_fund == "Greenhaven Road":
                st.write("""
                Greenhaven Road Capital is based in Connecticut.\n
                Portfolio Size: 15 Long Positions, handful of short\n
                AUM: $0.2 billion\n
                Greenhaven Road Capital is a long-biased, concentrated hedge fund and aims to identify small discarded, misunderstood, and esoteric stocks as opposed to large caps.
                """)
            
            elif selected_fund == "Cohen Capital Value":
                st.write("""
                Cohen Capital Value is based in Chicago, Illinois.\n
                Investment Horizon: 3-5 years\n
                Portfolio Size: 20 to 30 securities\n
                CCV has strived to make investments that meet the following characteristics (inverted for the short side): 1) Misunderstood, high-quality-assets, 2) At an inflection point, 3) With a margin of safety. The focus is on small caps. They are concentrated on finding companies that will pay them to own their stock in the form of dividends, buybacks, or other divestments. The fund sees multiple expansion as a bonus, but not required to drive performance.
                """)
            
            elif selected_fund == "Ensemble Capital Management":
                st.write("""
                Ensemble Capital Management is based in San Francisco, California.\n
                Investment Horizon: 3-5 years\n
                Portfolio Size: 15 to 30 securities\n
                AUM: $1.8 billion\n
                Ensemble Capital believes that investors are best served by owning a portfolio of companies, both emerging and established, that have significant growth opportunities and competitive advantages that allow them to sustain their profitability while returning capital to shareholders over time.
                """)

            # Fetch the available dates for the selected fund
            available_dates = self.fetch_available_dates(selected_fund)

            # Update the date range options based on the available dates
            start_quarter, end_quarter = st.sidebar.select_slider("Select Date Range", options=available_dates, value=(available_dates[0], available_dates[-1]))

            # Get all the quarters within the selected date range
            quarters_in_range = [date for date in available_dates if start_quarter <= date <= end_quarter]

            filtered_data = self.fetch_performance_data(selected_fund, start_quarter, end_quarter)
            self.display_line_graph(filtered_data)

            # Get the top 3 most discussed sectors
            top_sectors = self.get_top_sectors(selected_fund, start_quarter, end_quarter)

            st.write("The Specific Funds section enables you to dive deep into the performance and insights of individual hedge funds.")
            
            if top_sectors:
                st.write("**Top 3 most discussed sectors in the selected date range:**")
                for sector, count in top_sectors:
                    st.write(f"- {sector} ({count} mentions)")
            else:
                st.write("No sector data available for the selected date range.")

            st.header("Insight Extraction")
            insight_option = st.selectbox("Select an option to extract insights from the partner letters:", 
                                        ("Performance", "Strategy", "General Market Comments", "Ask Anything"))

            if insight_option == "Performance":
                st.write("<font color='blue'>**Please limit the date range to a maximum of 3 quarters for the best results.**</font>")
                performance_options = ['Key Contributors to Performance', 'Key Detractors from Performance', 'Portfolio Positioning and Adjustments']
                selected_performance_option = st.selectbox("Select a performance option", performance_options)
                
                if selected_performance_option:
                    result = self.handle_performance_button_click(selected_fund, quarters_in_range, selected_performance_option)
                    if result:
                        message_prompt, system_prompt, partner_letters, fund_names_dates = result
                        if st.button("Submit"):
                            print("Submit button clicked")
                            self.ai_response_generator.generate_response(message_prompt, system_prompt, partner_letters, fund_names_dates)
                    else:
                        print("Result is None")
                else:
                    print("No performance option selected")

            elif insight_option == "Strategy":
                st.write("Implement the logic for 'Strategy'")
                if st.button("Submit"):
                    print("Submit button clicked for Strategy")
                    # Implement the logic for 'Strategy' when the submit button is clicked
                    pass

            elif insight_option == "General Market Comments":
                st.write("Implement the logic for 'General Market Comments'")
                if st.button("Submit"):
                    print("Submit button clicked for General Market Comments")
                    # Implement the logic for 'General Market Comments' when the submit button is clicked
                    pass

            elif insight_option == "Ask Anything":
                st.write(f"The partner letters that match the selected fund ({selected_fund}) and date range ({start_quarter} to {end_quarter}) will be analyzed by GenAI.")
                
                user_input = st.text_input("Enter your question:")
                submit_button = st.button("Submit")
                
                if submit_button:
                    if user_input:
                        fund_names_dates = [f"{selected_fund} {quarter}" for quarter in quarters_in_range]
                        partner_letters = self.document_fetcher.fetch_partner_letters(fund_names_dates)
                        
                        system_prompt = f"""
                        You are an experienced investment analyst reviewing the quarterly partner letters from the hedge fund {selected_fund}.
                        The user has asked the following question: "{user_input}"
                        Your task is to analyze the provided partner letters and extract relevant insights to answer the user's question.
                        Provide a comprehensive response, citing specific examples from the letters to support your points.
                        After reviewing the relevant information, take a moment to organize your thoughts within <thinking></thinking> tags before presenting your final analysis to the user within <answer></answer> tags.
                        """
                        
                        print("Submit button clicked for Ask Anything")
                        self.ai_response_generator.generate_response(user_input, system_prompt, partner_letters, fund_names_dates)
                    else:
                        st.warning("Please enter a question before submitting.")

class VCDocumentFetcher:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_vc_partner_letters(self, fund_name, date):
        bucket_name = fund_name.split(" ")[0].lower()
        file_name = f"{bucket_name}/cleaned/{fund_name} {date}.txt"
        try:
            letter_content = self.aws_operations.fetch_object(file_name, "venturecapitalfunds")
            return letter_content
        except Exception as e:
            print(f"File not found: {file_name}")
            print(f"Error: {str(e)}")
            return None

class VCOpportunityScout:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_investments_data(self, selected_funds):
        investments_data = []
        for fund_name in selected_funds:
            bucket_name = fund_name.split(" ")[0].lower()
            file_name = f"{bucket_name}/{bucket_name}_investments.json"
            try:
                json_data = self.aws_operations.fetch_object(file_name, "venturecapitalfunds")
                investments_data.extend(json.loads(json_data))
            except Exception as e:
                print(f"File not found: {file_name}")
                print(f"Error: {str(e)}")
        return investments_data
    
    def run(self, fund_type, selected_funds):
        if not selected_funds:
            st.write("**Please select at least one fund in the side bar**")
            return
        
        if fund_type == "Venture Capital Funds":
            investments_data = self.fetch_investments_data(selected_funds)
            investment_types = sorted(set(item["Type of Investment"] for item in investments_data))
            selected_investment_type = st.selectbox("Select Type of Investment", ["All"] + investment_types)

            # Add the "Select Amount Invested" filter
            amount_invested_options = ["All", "<$1m", "$1m-$10m", ">$10m"]
            selected_amount_invested = st.selectbox("Select Amount Invested", amount_invested_options)

            # Add the "Fair Value of the Investment" filter
            fair_value_options = sorted(set(item["Fair Value of the Investment"] for item in investments_data))
            selected_fair_value = st.selectbox("Select Fair Value of the Investment", ["All"] + fair_value_options)

            if st.button("Submit"):
                filtered_data = investments_data

                if selected_investment_type != "All":
                    filtered_data = [item for item in filtered_data if item["Type of Investment"] == selected_investment_type]

                # Apply the "Select Amount Invested" filter
                if selected_amount_invested != "All":
                    if selected_amount_invested == "<$1m":
                        filtered_data = [item for item in filtered_data if float(item["Amount Invested"]) < 1]
                    elif selected_amount_invested == "$1m-$10m":
                        filtered_data = [item for item in filtered_data if 1 <= float(item["Amount Invested"]) <= 10]
                    else:  # ">$10m"
                        filtered_data = [item for item in filtered_data if float(item["Amount Invested"]) > 10]

                # Apply the "Fair Value of the Investment" filter
                if selected_fair_value != "All":
                    filtered_data = [item for item in filtered_data if item["Fair Value of the Investment"] == selected_fair_value]

                if filtered_data:
                    df = pd.DataFrame(filtered_data)
                    df["Amount Invested"] = df["Amount Invested"].apply(lambda x: "${:,.2f}".format(float(x)))
                    df = df[["Fund", "Date", "Company", "Type of Investment", "Amount Invested", "Date invested", "Fair Value of the Investment", "Summary"]]
                    
                    df.reset_index(drop=True, inplace=True)

                    # Add the message to inform the user about double-clicking on a cell
                    st.write("**Double-click on a cell to see the full text.**")

                    st.dataframe(df)
                else:
                    st.write("No data found for the selected filters.")
        else:
            st.write("This feature is not available for the selected fund type.")  
            
class SpecificVCFundsSection:
    def __init__(self, aws_operations, ai_response_generator, vc_document_fetcher):
        self.aws_operations = aws_operations
        self.ai_response_generator = ai_response_generator
        self.vc_document_fetcher = vc_document_fetcher

    def fetch_performance_data(self, selected_fund):
        # Fetch the performance data from the JSON file in the S3 bucket
        performance_data = self.aws_operations.fetch_object("vc_performance_insights.json", "venturecapitalfunds")
        performance_data = json.loads(performance_data)

        # Filter the performance data based on the selected fund
        filtered_data = [obj for obj in performance_data if obj['Fund Name'] == selected_fund]

        return filtered_data

    def display_performance_table(self, filtered_data):
        # Extract the relevant columns from the filtered data for the performance table
        table_data = [
            {
                'Date': obj['Date'],
                'Net IRR': obj['Net IRR'],
                'Percentage Capital Commitments Called': obj['Percentage Capital Commitments Called']
            }
            for obj in filtered_data
        ]

        # Create a DataFrame from the table data
        df = pd.DataFrame(table_data)

        st.table(df)

    def display_selected_text(self, filtered_data, selected_option):
        # Extract the text based on the selected option
        text = filtered_data[0].get(selected_option, '')

        st.write(text)

    def generate_vc_response(self, prompt, system_prompt, partner_letter, fund_name, date):
        client = anthropic.Anthropic(api_key=self.ai_response_generator.api_key)
        
        # Create XML tags for the document
        fund_name = fund_name.replace(" ", "").replace(",", "")
        quarter = date.split(" ")[0].lower()
        year = date.split(" ")[1]
        tag = f"<{fund_name}_{year}_{quarter}>"
        tagged_letter = f"{tag}\n{partner_letter}\n</{fund_name}_{year}_{quarter}>"
        
        message = client.messages.create(
            model=CLAUDE_HAIKU,
            max_tokens=2000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{tagged_letter}\n\n{prompt}"
                        }
                    ]
                }
            ]
        )
        
        raw_text = message.content
        answer = raw_text[0].text
        
        # Extract the content within <answer></answer> tags if present
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        st.write(answer)

    def handle_ask_anything(self, selected_fund, filtered_data):
        user_input = st.text_input("Enter your question:")

        if user_input:
            date = filtered_data[0]['Date']
            partner_letter = self.vc_document_fetcher.fetch_vc_partner_letters(selected_fund, date)

            if partner_letter:
                system_prompt = f"""
                You are an experienced venture capital analyst reviewing the quarterly letter from the VC fund {selected_fund}.
                Venture capitalists invest in early-stage, high-growth potential companies with the goal of generating significant returns for their investors. They write quarterly letters to provide updates on the fund's performance, portfolio companies, and market insights to their limited partners (investors).
                The user will ask a question about the provided quarterly letter and I want you to do the best job at answering it.
                Your task is to analyze the provided partner letter and extract relevant insights to answer the user's question.
                Provide a comprehensive response, citing specific examples from the letter to support your points.
                After reviewing the relevant information, take a moment to organize your thoughts and present your final analysis to the user.
                """

                message_prompt = f"""
                This is the user's question:
                <user_input>
                {user_input}
                </user_input>

                Please provide a detailed response based on the information in the quarterly letter from {selected_fund}. Make sure you cite your sources correctly and provide a well-structured answer.
                """

                self.generate_vc_response(message_prompt, system_prompt, partner_letter, selected_fund, date)
            else:
                st.write("Partner letter not found for the selected fund and date.")

    def run(self, selected_fund):
        if selected_fund:
            st.title(selected_fund)

            filtered_data = self.fetch_performance_data(selected_fund)

            if filtered_data:
                self.display_performance_table(filtered_data)

                options = [
                    "Commentary on Fund Performance",
                    "Key Contributors to Performance",
                    "Key Detractors from Performance",
                    "Portfolio Positioning and Adjustments",
                    "Ask Anything"
                ]
                selected_option = st.selectbox("Select an option", options)

                if selected_option == "Ask Anything":
                    self.handle_ask_anything(selected_fund, filtered_data)
                elif st.button("Submit"):
                    self.display_selected_text(filtered_data, selected_option)
            else:
                st.write("No performance data found for the selected fund.")

class SourcesSection:
    def __init__(self, aws_operations):
        self.aws_operations = aws_operations

    def fetch_fund_names(self, bucket_name, fund_info_path):
        # Fetch the JSON file from S3
        json_data = self.aws_operations.fetch_object(fund_info_path, bucket_name)
        fund_info_data = json.loads(json_data)

        # Get unique fund names
        fund_names = set(obj['Fund Name'] for obj in fund_info_data)

        return list(fund_names)

    def run(self):
        source_option = st.radio(
            "Select a source option",
            ("Hedge Fund Partner Letters", "Podcasts", "VC Documents")
        )

        if source_option == "Hedge Fund Partner Letters":
            bucket_name = "hedgefunds"
            fund_info_path = "hedgefund_general_insights.json"
            fund_names = self.fetch_fund_names(bucket_name, fund_info_path)

            if len(fund_names) > 0:
                # Create a one-column DataFrame with fund names
                fund_data = pd.DataFrame({"Fund Name": fund_names})

                # Display the DataFrame
                st.write("Fund Names:")
                st.dataframe(fund_data)
            else:
                st.write("No fund names found.")

        elif source_option == "Podcasts":
            st.write("Implement the logic for 'Podcasts'")
            # Add your implementation here

        elif source_option == "VC Documents":
            st.write("Implement the logic for 'VC Documents'")
            # Add your implementation here

        # File upload section
        st.write("Upload your own documents:")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            # Save the uploaded file to S3
            self.aws_operations.upload_object(uploaded_file.getvalue(), uploaded_file.name)
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")

def main():
    st.set_page_config(layout="wide")
    aws_operations = AWSOperations()
    ai_response_generator = AIResponseGenerator(ANTHROPIC_API_KEY)
    document_fetcher = DocumentFetcher(aws_operations)
    market_mood_monitor = MarketMoodMonitor(aws_operations, ai_response_generator, "hedgefund_general_insights.json", document_fetcher)
    sources_section = SourcesSection(aws_operations)
    performance_pulse = PerformancePulse(aws_operations, ai_response_generator, document_fetcher)
    specific_funds_section = SpecificFundsSection(aws_operations, ai_response_generator, document_fetcher)
    vc_document_fetcher = VCDocumentFetcher(aws_operations)
    specific_vc_funds_section = SpecificVCFundsSection(aws_operations, ai_response_generator, vc_document_fetcher)
    vc_opportunity_scout = VCOpportunityScout(aws_operations) 

    selected_option = st.sidebar.radio(
        "Navigation",
        ("Home", "Bird's-Eye View", "Specific Funds", "Sources")
    )

    if selected_option == "Home":
        st.markdown("<h1 style='text-align: center; color: blue;'>🚀 Welcome to wybe.ai!</h1>", unsafe_allow_html=True)
        st.markdown("""
        <div style='text-align: center;'>
            <br>
            <p>Welcome to wybe.ai, a powerful Insight Extractor designed to help asset allocators gain quick and valuable insights from the funds they are invested in. This application leverages GenAI technology to automate manual tasks and save an asset allocator tons of time so they can focus on what they do best: making investment decisions.</p>
            <br>
            <p>Please note that this is a demo version, and there is significant room for improvement in areas such as optimizing prompts, using agents, connecting to the internet for live data, and enhancing document search capabilities within the database. We appreciate your understanding and look forward to your feedback as we continue to refine and expand the functionality of this tool!</p>
        </div>
        """, unsafe_allow_html=True)

    elif selected_option == "Bird's-Eye View":
        fund_type = st.sidebar.radio(
            "Select a Fund Type",
            ("Hedge Funds", "Venture Capital Funds", "Private Equity Funds"),
            key="main_fund_type"
        )

        if fund_type == "Venture Capital Funds":
            asset_allocator_option = "Opportunity Scout"
        else:
            asset_allocator_option = st.sidebar.selectbox(
                "Select an option",
                ("Opportunity Scout", "Performance Pulse", "Strategy Scanner", "Market Mood Monitor", "Compliance Compass")
            )

        bucket_name = get_bucket_name(fund_type)

        if bucket_name:
            if fund_type == "Hedge Funds":
                fund_insights_path = "hedgefund_general_insights.json"
            elif fund_type == "Venture Capital Funds":
                fund_insights_path = "vc_performance_insights.json"
            else:
                fund_insights_path = None

            if fund_insights_path:
                selected_funds = select_funds(aws_operations, bucket_name, fund_insights_path)
                formatted_selected_funds = format_fund_names(selected_funds, fund_type)
            else:
                formatted_selected_funds = []
        else:
            formatted_selected_funds = []

        if asset_allocator_option == "Opportunity Scout":
            if fund_type == "Hedge Funds":
                opportunity_scout = OpportunityScout(aws_operations, bucket_name)
                st.title("Opportunity Scout")
                st.subheader(f"Extract Key Insights about your {fund_type} Performance")
                st.write("The Opportunity Scout allows you to filter and analyze companies from the selected hedge funds based on sectors, date range, and investment status (pitched or exited).")
                opportunity_scout.run(fund_type, formatted_selected_funds)
            elif fund_type == "Venture Capital Funds":
                st.title("Opportunity Scout")
                st.subheader(f"Extract Key Insights about your {fund_type} Performance")
                st.write("The Opportunity Scout allows you to filter and analyze companies pitched by the selected VC funds")
                vc_opportunity_scout.run(fund_type, formatted_selected_funds)
            else:
                st.write("This feature is not available for the selected fund type.")
        else:
            st.write("Implementation will come soon.")

    elif selected_option == "Specific Funds":
        fund_type = st.sidebar.radio(
            "Select a Fund Type",
            ("Hedge Funds", "Venture Capital Funds", "Private Equity Funds")
        )

        bucket_name = get_bucket_name(fund_type)
        
        if bucket_name:
            fund_info_path = "hedgefund_general_insights.json" if fund_type == "Hedge Funds" else "vc_performance_insights.json"
            fund_names = fetch_fund_names(aws_operations, bucket_name, fund_info_path)
            selected_fund = st.sidebar.selectbox(f"Select a {fund_type}", fund_names)
            
            if fund_type == "Hedge Funds":
                specific_funds_section.run(selected_fund)
            elif fund_type == "Venture Capital Funds":
                specific_vc_funds_section.run(selected_fund)
            else:  # Private Equity Funds
                st.write("Logic for PE funds will come soon!")
            
    elif selected_option == "Sources":
        sources_section.run()

def get_bucket_name(fund_type):
    bucket_map = {
        "Hedge Funds": "hedgefunds",
        "Venture Capital Funds": "venturecapitalfunds"
    }
    return bucket_map.get(fund_type)

def select_funds(aws_operations, bucket_name, fund_insights_path):
    fund_names = fetch_fund_names(aws_operations, bucket_name, fund_insights_path)
    fund_names_list = list(fund_names)
    selected_funds = st.sidebar.multiselect("Select Funds", fund_names_list)
    return selected_funds

def format_fund_names(fund_names, fund_type):
    if fund_type == "Hedge Funds":
        return [fund.lower().replace(" ", "") for fund in fund_names]
    else:
        return fund_names

def display_selected_funds(selected_funds):
    if selected_funds:
        st.write("Selected Funds:")
        for fund in selected_funds:
            st.write(fund)
    else:
        st.write("No funds selected.")

if __name__ == '__main__':
    main()