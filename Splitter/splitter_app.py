import streamlit as st
import openai
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to communicate with OpenAI model
def llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message['content']

# Parsing functions for different document types
def parse_html(html_content):
    prompt = f"""
    You are helping convert an HTML document into a PDF. Here’s a sample HTML page content:
    {html_content}

    Please extract the content as if you’re creating PDF page entries. Use the format:
    [Page X: Title, Header, Paragraphs, Image descriptions...]
    Also, specify the type of content (Text/Image) for each entry.
    """
    return llm(prompt)

def parse_excel(excel_content):
    prompt = f"""
    You’re converting an Excel spreadsheet into a PDF. Here’s the spreadsheet data:
    {excel_content}

    Please organize this content as if it’s being formatted for a PDF, with each sheet as a new page. Use this format:
    [Page X: Table Title, Column Headers, Row Entries, Visual Elements...]
    Indicate whether each element is "Text" or "Image."
    """
    return llm(prompt)

def parse_word(word_content):
    prompt = f"""
    You are converting a Word document into a PDF. Here’s the document content:
    {word_content}

    Please organize this content as if it’s formatted for a PDF, with each section as a new page. Use this format:
    [Page X: Title, Header, Paragraphs, Images...]
    Specify each content type as "Text" or "Image."
    """
    return llm(prompt)

def parse_pdf(pdf_text):
    prompt = f"""
    You are helping convert a PDF document into a formatted PDF for presentation. Here’s the content extracted from the PDF:
    {pdf_text}

    Please organize the content with titles, headers, paragraphs, and images if present. Format each entry as follows:
    [Page X: Title, Header, Paragraphs, Image Descriptions...]
    Specify whether the content is "Text" or "Image."
    """
    return llm(prompt)

# Compile PDF content and structure
def compile_pdf(parsed_pages):
    pdf_structure = []
    for page in parsed_pages:
        pdf_structure.append(page)
    return pdf_structure

# Calculate text-to-image ratios
def calculate_content_ratios(pdf_content):
    prompt = f"""
    Based on the content extracted, calculate the percentage of text and images in the PDF. 
    For each page:
    - Count words in all text entries.
    - Count the number of images.
    Then, provide the overall percentages of text vs. images for the entire document.
    
    Here’s the content for analysis:
    {pdf_content}
    """
    return llm(prompt)

# Function to plot bar graph
def plot_bar_graph(text_percentage, image_percentage):
    categories = ['Text', 'Image']
    percentages = [text_percentage, image_percentage]
    
    fig, ax = plt.subplots()
    ax.bar(categories, percentages, color=['blue', 'green'])
    
    ax.set_xlabel('Content Type')
    ax.set_ylabel('Percentage')
    ax.set_title('Text-to-Image Ratio')
    
    # Display the plot
    st.pyplot(fig)

# Main function to process file contents
def convert_to_pdf(file_contents):
    parsed_pages = []
    
    for content in file_contents:
        if content['type'] == 'html':
            parsed_pages.append(parse_html(content['data']))
        elif content['type'] == 'excel':
            parsed_pages.append(parse_excel(content['data']))
        elif content['type'] == 'word':
            parsed_pages.append(parse_word(content['data']))
        elif content['type'] == 'pdf':
            parsed_pages.append(parse_pdf(content['data']))
    
    pdf_content = compile_pdf(parsed_pages)
    content_ratios = calculate_content_ratios(pdf_content)
    
    return {
        'pdf_content': pdf_content,
        'content_ratios': content_ratios
    }

# Streamlit app setup
st.title("Splitter: Multi-File to PDF Converter")
st.write("Upload HTML, Excel, Word, or PDF files to convert them into a combined PDF and get text-image analysis.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload HTML, Excel, Word, or PDF files (Maximum 5 files)",
    type=["html", "xls", "xlsx", "docx", "pdf"],
    accept_multiple_files=True
)

# Check if files are uploaded
if uploaded_files:
    if len(uploaded_files) > 5:
        st.error("Please upload no more than 5 files.")
    else:
        file_contents = {
            'html': [],
            'excel': [],
            'word': [],
            'pdf': []
        }
        
        # Categorize the uploaded files based on type
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.type
            file_data = uploaded_file.read()
            
            if file_type == "text/html":
                file_contents['html'].append(file_data.decode("utf-8"))
            elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                file_contents['excel'].append(file_data)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                file_contents['word'].append(file_data)
            elif file_type == "application/pdf":
                file_contents['pdf'].append(file_data.decode("utf-8", errors='ignore'))  # Assuming you can extract PDF text directly

        if st.button("Convert and Analyze"):
            # Convert to PDF and analyze
            processed_files = []
            for file_type, files in file_contents.items():
                for file in files:
                    processed_files.append({"type": file_type, "data": file})

            output = convert_to_pdf(processed_files)
            
            # Display PDF content and analysis
            st.write("PDF Content:", output['pdf_content'])
            st.write("Text to Image Ratio:", output['content_ratios'])
            
            # Extract text and image percentages from the output and plot the bar chart
            try:
                # Parse the content ratio into text and image percentages
                ratios = output['content_ratios'].split("\n")
                text_percentage = float(ratios[0].split(":")[1].strip().replace('%', ''))
                image_percentage = float(ratios[1].split(":")[1].strip().replace('%', ''))
                
                # Plot the bar graph
                plot_bar_graph(text_percentage, image_percentage)
                
            except Exception as e:
                st.error(f"Error in processing content ratios: {str(e)}")
else:
    st.write("Upload files to begin conversion.")