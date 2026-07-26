import streamlit as st
import os
import json
from datetime import datetime
import PyPDF2
from docx import Document
from openai import OpenAI
from typing import List, Dict, Tuple
import pandas as pd
from io import BytesIO

# Configure Streamlit
st.set_page_config(
    page_title="Resume Analysis Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .match-score {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .match-high {
        background-color: #d4edda;
        color: #155724;
    }
    .match-medium {
        background-color: #fff3cd;
        color: #856404;
    }
    .match-low {
        background-color: #f8d7da;
        color: #721c24;
    }
    .report-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

class ResumeAnalyzer:
    """Handles resume extraction and analysis using OpenAI API"""
    
    def __init__(self, api_key: str):
        """Initialize the analyzer with OpenAI API key"""
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
    
    def extract_text_from_pdf(self, file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_text_from_docx(self, file) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file)
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_text_from_txt(self, file) -> str:
        """Extract text from TXT file"""
        try:
            return file.read().decode('utf-8')
        except Exception as e:
            raise Exception(f"Error reading TXT: {str(e)}")
    
    def extract_resume_text(self, file) -> str:
        """Extract text from various resume formats"""
        file_extension = file.name.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            return self.extract_text_from_pdf(file)
        elif file_extension == 'docx':
            return self.extract_text_from_docx(file)
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def analyze_resume(self, resume_text: str, job_description: str) -> Dict:
        """Analyze resume against job description using OpenAI"""
        try:
            prompt = f"""
            Analyze the following resume against the job description and provide:
            1. A matching percentage (0-100)
            2. Key skills match
            3. Experience alignment
            4. Education match
            5. Top 5 strengths for this role
            6. Areas for improvement
            7. Overall recommendation
            
            Please format the response as JSON with the following structure:
            {{
                "matching_percentage": <number>,
                "key_skills_match": {{"matched": [list], "missing": [list]}},
                "experience_alignment": "<description>",
                "education_match": "<description>",
                "top_strengths": [<list of 5 items>],
                "areas_for_improvement": [<list of items>],
                "recommendation": "<description>",
                "summary": "<brief summary>"
            }}
            
            Resume:
            {resume_text}
            
            Job Description:
            {job_description}
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert resume analyst and recruiter. Provide detailed analysis in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from the response
            try:
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx != -1 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except json.JSONDecodeError:
                # If JSON parsing fails, create a structured response
                analysis = {
                    "matching_percentage": 50,
                    "key_skills_match": {"matched": [], "missing": []},
                    "experience_alignment": response_text[:500],
                    "education_match": "Unable to parse",
                    "top_strengths": [],
                    "areas_for_improvement": [],
                    "recommendation": response_text,
                    "summary": response_text[:200]
                }
            
            return analysis
        except Exception as e:
            raise Exception(f"Error analyzing resume: {str(e)}")
    
    def generate_report(self, analyses: List[Dict], resume_names: List[str]) -> str:
        """Generate a comprehensive report from multiple resume analyses"""
        report = f"""
RESUME ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}
Total Resumes Analyzed: {len(analyses)}
Average Matching Score: {sum(a.get('matching_percentage', 0) for a in analyses) / len(analyses):.1f}%
Best Match: {max(analyses, key=lambda x: x.get('matching_percentage', 0)).get('matching_percentage', 0):.1f}%

DETAILED ANALYSIS
{'-'*80}
"""
        
        for idx, (analysis, name) in enumerate(zip(analyses, resume_names), 1):
            report += f"\n\nCANDIDATE {idx}: {name}\n"
            report += "-" * 80 + "\n"
            report += f"Matching Percentage: {analysis.get('matching_percentage', 0)}%\n"
            report += f"Summary: {analysis.get('summary', 'N/A')}\n\n"
            
            # Skills Match
            skills = analysis.get('key_skills_match', {})
            if skills:
                report += f"Matched Skills: {', '.join(skills.get('matched', []))}\n"
                report += f"Missing Skills: {', '.join(skills.get('missing', []))}\n\n"
            
            # Experience & Education
            report += f"Experience Alignment:\n{analysis.get('experience_alignment', 'N/A')}\n\n"
            report += f"Education Match:\n{analysis.get('education_match', 'N/A')}\n\n"
            
            # Strengths
            strengths = analysis.get('top_strengths', [])
            if strengths:
                report += "Top Strengths:\n"
                for strength in strengths:
                    report += f"  • {strength}\n"
                report += "\n"
            
            # Improvements
            improvements = analysis.get('areas_for_improvement', [])
            if improvements:
                report += "Areas for Improvement:\n"
                for improvement in improvements:
                    report += f"  • {improvement}\n"
                report += "\n"
            
            # Recommendation
            report += f"Recommendation: {analysis.get('recommendation', 'N/A')}\n"
            report += "\n"
        
        return report

def get_match_color(percentage: float) -> str:
    """Determine color class based on matching percentage"""
    if percentage >= 75:
        return "match-high"
    elif percentage >= 50:
        return "match-medium"
    else:
        return "match-low"

def main():
    # Title
    st.markdown('<div class="main-header">📄 Resume Analysis Tool</div>', unsafe_allow_html=True)
    st.markdown("Analyze multiple resumes against a job description using AI-powered insights")
    
    # Sidebar for API Key and Job Description
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key. This is required for resume analysis."
        )
        
        if not api_key:
            st.warning("Please enter your OpenAI API key to proceed.")
            st.info("Get your API key from: https://platform.openai.com/api-keys")
            return
        
        # Job Description
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=300,
            help="Enter the complete job description for resume matching"
        )
        
        if not job_description:
            st.warning("Please enter a job description to analyze resumes.")
            return
    
    # Main content area
    st.header("Upload Resumes")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose resume files (PDF, DOCX, or TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="Upload one or more resume files for analysis"
    )
    
    if not uploaded_files:
        st.info("👆 Please upload at least one resume file to get started")
        return
    
    st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
    
    # Analyze button
    if st.button("🚀 Analyze Resumes", type="primary", use_container_width=True):
        
        try:
            # Initialize analyzer
            analyzer = ResumeAnalyzer(api_key)
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            analyses = []
            resume_names = []
            
            # Process each resume
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing resume {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")
                progress_bar.progress((idx + 1) / len(uploaded_files))
                
                # Extract text
                resume_text = analyzer.extract_resume_text(uploaded_file)
                
                # Analyze resume
                analysis = analyzer.analyze_resume(resume_text, job_description)
                analyses.append(analysis)
                resume_names.append(uploaded_file.name)
            
            status_text.text("✅ Analysis complete!")
            progress_bar.empty()
            
            # Display results
            st.header("Analysis Results")
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_match = sum(a.get('matching_percentage', 0) for a in analyses) / len(analyses)
                st.metric("Average Match %", f"{avg_match:.1f}%")
            
            with col2:
                best_match = max(analyses, key=lambda x: x.get('matching_percentage', 0))
                best_idx = analyses.index(best_match)
                st.metric("Best Match", f"{best_match.get('matching_percentage', 0):.1f}%", resume_names[best_idx])
            
            with col3:
                st.metric("Resumes Analyzed", len(analyses))
            
            # Detailed results for each resume
            st.header("Detailed Analysis")
            
            tabs = st.tabs([name.split('.')[0] for name in resume_names])
            
            for tab, analysis, name in zip(tabs, analyses, resume_names):
                with tab:
                    # Matching percentage with color
                    match_pct = analysis.get('matching_percentage', 0)
                    color_class = get_match_color(match_pct)
                    st.markdown(
                        f'<div class="match-score {color_class}">{match_pct}% Match</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Summary
                    st.markdown('<div class="report-section">', unsafe_allow_html=True)
                    st.write("**Summary:**")
                    st.write(analysis.get('summary', 'N/A'))
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Skills
                    st.subheader("Skills Analysis")
                    skills = analysis.get('key_skills_match', {})
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        matched_skills = skills.get('matched', [])
                        st.write("✅ **Matched Skills:**")
                        if matched_skills:
                            for skill in matched_skills:
                                st.write(f"  • {skill}")
                        else:
                            st.write("No matched skills found")
                    
                    with col2:
                        missing_skills = skills.get('missing', [])
                        st.write("❌ **Missing Skills:**")
                        if missing_skills:
                            for skill in missing_skills:
                                st.write(f"  • {skill}")
                        else:
                            st.write("No missing skills")
                    
                    # Experience and Education
                    st.subheader("Experience & Education")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Experience Alignment:**")
                        st.write(analysis.get('experience_alignment', 'N/A'))
                    
                    with col2:
                        st.write("**Education Match:**")
                        st.write(analysis.get('education_match', 'N/A'))
                    
                    # Strengths
                    st.subheader("Top Strengths")
                    strengths = analysis.get('top_strengths', [])
                    if strengths:
                        for i, strength in enumerate(strengths, 1):
                            st.write(f"{i}. {strength}")
                    else:
                        st.write("No strengths data available")
                    
                    # Areas for Improvement
                    st.subheader("Areas for Improvement")
                    improvements = analysis.get('areas_for_improvement', [])
                    if improvements:
                        for improvement in improvements:
                            st.write(f"• {improvement}")
                    else:
                        st.write("No improvement areas identified")
                    
                    # Recommendation
                    st.subheader("Recommendation")
                    st.info(analysis.get('recommendation', 'N/A'))
            
            # Comparison Table
            st.header("Quick Comparison")
            
            comparison_data = []
            for name, analysis in zip(resume_names, analyses):
                comparison_data.append({
                    "Resume": name,
                    "Match %": f"{analysis.get('matching_percentage', 0):.1f}%",
                    "Matched Skills": len(analysis.get('key_skills_match', {}).get('matched', [])),
                    "Missing Skills": len(analysis.get('key_skills_match', {}).get('missing', []))
                })
            
            df = pd.DataFrame(comparison_data)
            st.dataframe(df, use_container_width=True)
            
            # Download Report
            st.header("📥 Download Report")
            
            report = analyzer.generate_report(analyses, resume_names)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # TXT Report
                st.download_button(
                    label="📄 Download as TXT",
                    data=report,
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col2:
                # JSON Report
                json_report = {
                    "generated_at": datetime.now().isoformat(),
                    "total_resumes": len(analyses),
                    "job_description": job_description,
                    "analyses": analyses,
                    "resume_names": resume_names,
                    "average_match": sum(a.get('matching_percentage', 0) for a in analyses) / len(analyses)
                }
                st.download_button(
                    label="📊 Download as JSON",
                    data=json.dumps(json_report, indent=2),
                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.info("Please check your API key and try again.")

if __name__ == "__main__":
    main()
