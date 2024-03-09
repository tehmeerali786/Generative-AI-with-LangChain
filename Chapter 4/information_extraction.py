from dotenv import load_dotenv
load_dotenv()

from typing import Optional 

from langchain.chains import create_extraction_chain_pydantic 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from pydantic import BaseModel, Field 

class Experience(BaseModel):
    # the title doesn't seem to help at all
    start_date: Optional[str] = Field(description="When the job or study started.")
    end_date: Optional[str] = Field(description="When the job or study ended.")
    description: Optional[str] = Field(description="What the job or study entailed.")
    country: Optional[str] = Field(description="The country of the institution.")
    
class Study(Experience):
    degree: Optional[str] = Field(description="The degree obtained or expected.")
    institution: Optional[str] = Field(
        description="The university, college, or educational institution visited."
        )
    country: Optional[str] = Field(description="The country of the institution.")
    grade: Optional[str] = Field(description="The grades achieved or expected.")
    
class WorkExperience(Study):
    company: str = Field(description="The company name of the work experience.")
    job_title: Optional[str] = Field(description="The job title.")
    
class Resume(BaseModel):
    first_name: Optional[str] = Field(description="The first name of the person.")
    last_name: Optional[str] = Field(description="The last name of the person.")
    linkedin_url: Optional[str] = Field(
        description="A study that the person completed or is in progress of completing."
    )
    email_address: Optional[str] = Field(description="The email address of the person.")
    nationality: Optional[str] = Field(description="The nationality of the person.")
    skill: Optional[str] = Field(description="A skill listed or mentioned in a description.")
    study: Optional[Study] = Field(
        description="A study that the person completed or is in progress of completing."
        )
    work_experience: Optional[WorkExperience] = Field(description="A hobby or recreational activity of the person.")
    

def parse_cv(pdf_file_path: str) -> str:
    """Parse a resume.
    Not totally sure about the return type: is it list[Resume]?
    
    """
    pdf_loader = PyPDFLoader(pdf_file_path)
    docs = pdf_loader.load_and_split()
    # please not the function calling is not enabled for all models!
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    chain = create_extraction_chain_pydantic(pydantic_schema=Resume, llm=llm)
    return chain.run(docs)

if __name__ == "__main__":
    print(parse_cv(
        pdf_file_path="openresume-resume.pdf"
    ))    
    
    