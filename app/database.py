from sqlalchemy import Column, String, Float, DateTime, create_engine, Boolean
# define SQL structure/table
from sqlalchemy.orm import declarative_base, sessionmaker

from datetime import datetime, timezone


# ----- env -----
import os
from dotenv import load_dotenv
load_dotenv()

# DB
DB_URL = os.getenv('DATABASE_URL')
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# allows to define db models
Base = declarative_base()

class ImageRecord(Base):
    __tablename__ = "image_records"

    # Define the columns for the table
    url = Column(String, primary_key=True, unique=True)
    image_type = Column(String, nullable=False)
    predicted_class = Column(String, nullable=True)
    status = Column(String, nullable=True)
    confidence_level = Column(Float, nullable=True)
    job_id = Column(String, nullable=True)
    folder_location = Column(String, nullable=True)
    local_filename = Column(String, nullable=True)
    re_label = Column(Boolean, default=False, nullable=False)           # a flag that says the image needs to be reclassified and retrained
    admin_reviewed = Column(Boolean, default=False, nullable=False)     # a flag that says the review is done 
    local_url = Column(String, nullable=True)
    prediction_model_version = Column(String, nullable=True)
    upload_file_name = Column(String, nullable=True)  
    
    # This field automatically sets the creation timestamp
    datetime_added = Column(DateTime, default=datetime.now(timezone.utc), nullable=False)
    
    
def init_db():
    """
    Initializes the database by creating all tables defined in the models.
    """
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialization complete.")
    
def get_db():
    """
    This method allows to start and create a Session, which will be closed once the calloing method has executed. 
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()