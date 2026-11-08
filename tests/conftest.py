import os

os.environ["USE_SAMPLE_DATA"] = "1"   # tests are written against the small sample catalog
os.environ.pop("GROQ_API_KEY", None)   # no network LLM calls in tests
