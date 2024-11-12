import os
import streamlit as st
from fpdf import FPDF
import openai
import matplotlib.pyplot as plt
import fitz  # PyMuPDF
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to communicate with OpenAI model
def llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Consider using gpt-3.5-turbo to reduce cost
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message['content']
    except openai.error.RateLimitError:
        st.error("Rate limit reached. Please wait and try again later.")
        return None
    except openai.error.InvalidRequestError as e:
        st.error(f"API request error: {e}")
        return None

# Function to parse PDF file and extract text
def parse_pdf(pdf_file):
    pdf_data = io.BytesIO(pdf_file.read())
    doc = fitz.open("pdf", pdf_data)  # Open PDF using BytesIO object
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page of the PDF
    return text

# Folder to save generated PDFs
SAVE_PATH = r"C:\Users\fahee\OneDrive\Desktop\mini project for Mphasis\Complete-Langchain-Tutorials\Splitter"

# Create the folder if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

# Function to generate a PDF from text
def generate_pdf(content, filename):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Combined PDF Document", ln=True, align='C')  # Add a title
    
    pdf.ln(10)  # Line break
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(200, 10, txt=content)
    
    pdf_output_path = os.path.join(SAVE_PATH, filename)
    pdf.output(pdf_output_path)
    return pdf_output_path

# Function to analyze PDF (dummy implementation)
def analyze_pdf(pdf_file_path):
    try:
        # Open the generated PDF
        doc = fitz.open(pdf_file_path)
        total_pages = doc.page_count
        total_text_area = 0
        total_image_area = 0

        for page_num in range(total_pages):
            page = doc[page_num]
            page_area = page.rect.width * page.rect.height  # Total area of the page

            # Get text blocks and calculate their area
            text_blocks = page.get_text("blocks")
            for block in text_blocks:
                x0, y0, x1, y1, *_ = block  # Block coordinates
                block_area = (x1 - x0) * (y1 - y0)
                total_text_area += block_area

            # Get images and calculate their area
            image_blocks = page.get_images(full=True)
            for img in image_blocks:
                xref = img[0]  # Image reference
                img_rect = page.get_image_rects(xref)[0]  # Get image rectangle
                img_area = img_rect.width * img_rect.height
                total_image_area += img_area

        # Calculate percentages
        total_content_area = total_text_area + total_image_area
        if total_content_area == 0:
            st.warning("No text or images detected in the PDF.")
            return 0, 0  # No content, so return 0 for both

        text_percentage = (total_text_area / total_content_area) * 100
        image_percentage = (total_image_area / total_content_area) * 100

        return round(text_percentage, 2), round(image_percentage, 2)

    except Exception as e:
        st.error(f"Error analyzing PDF: {str(e)}")
        return None, None

# Function to plot bar graph
def plot_bar_graph(text_percentage, image_percentage):
    categories = ['Text', 'Image']
    percentages = [text_percentage, image_percentage]
    
    plt.bar(categories, percentages, color=['blue', 'orange'])
    plt.xlabel('Content Type')
    plt.ylabel('Percentage')
    plt.title('Text to Image Content Analysis')
    
    # Display the plot in Streamlit
    st.pyplot(plt)

# Streamlit app setup
st.title("PDF Splitter and Analyzer")
st.write("Upload a PDF file to convert it into a formatted PDF and get text-image analysis.")

# File uploader
uploaded_files = st.file_uploader(
    "Upload PDF files (Maximum 5 files)",
    type=["pdf"],
    accept_multiple_files=True
)

# Check if files are uploaded
if uploaded_files:
    # Step 1: Extract and combine text from all PDFs
    combined_text = ""
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            pdf_text = parse_pdf(uploaded_file)
            if pdf_text:
                combined_text += pdf_text + "\n\n"  # Separate text from each PDF with line breaks

    # Step 2: Call LLM with the combined text if extraction was successful
    if combined_text:
        prompt = f"""
        You are helping convert a combined PDF document into a formatted version for presentation. Here’s the combined extracted text from all uploaded PDFs:
        {combined_text}

        Please organize the content with titles, headers, and paragraphs suitable for a professional PDF.
        """
        
        pdf_content = llm(prompt)  # Call the LLM once with the combined content

        if pdf_content:
            # Step 3: Generate the formatted PDF
            pdf_filename = "combined_output.pdf"
            pdf_path = generate_pdf(pdf_content, pdf_filename)
            st.write(f"PDF generated and saved at: {pdf_path}")

            # Step 4: Perform text-image analysis on the generated PDF
            text_percentage, image_percentage = analyze_pdf(pdf_path)

            if text_percentage is not None and image_percentage is not None:
                st.write(f"Text: {text_percentage}% | Image: {image_percentage}%")
                plot_bar_graph(text_percentage, image_percentage)

            st.success("PDF conversion and analysis complete!")
        else:
            st.error("Failed to generate formatted content using LLM.")
    else:
        st.error("No text extracted from the uploaded PDFs.")
