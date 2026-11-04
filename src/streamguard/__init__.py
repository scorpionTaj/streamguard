"""StreamGuard — real-time fraud detection with an online feature store.

Pipeline: replay producer → Redpanda → Spark Structured Streaming feature job
→ Redis online store → FastAPI scorer → Streamlit alerts.
See PLAN.md for the full spec; milestone status at the top of each module.
"""

__version__ = "0.1.0"
