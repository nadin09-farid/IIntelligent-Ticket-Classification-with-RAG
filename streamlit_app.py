"""
Streamlit demo UI for the Intelligent Ticket Classification & RAG system.

This is a thin frontend: it sends the subject/body to the FastAPI backend's
/tickets/process endpoint and displays the results nicely. It does NOT run
the pipeline itself — the FastAPI server (api.py) must be running separately.

Run:
    1. In one terminal: uvicorn api:app --reload --port 8000
    2. In another terminal: streamlit run streamlit_app.py
"""
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/tickets/process"

st.set_page_config(page_title="Ticket Classification & RAG", page_icon="🎫", layout="wide")

st.title("🎫 Intelligent Ticket Classification & RAG")
st.caption("Classifies a support ticket, retrieves similar past cases, and drafts a response.")

with st.form("ticket_form"):
    subject = st.text_input("Ticket subject", placeholder="e.g. Cannot log in")
    body = st.text_area(
        "Ticket body",
        placeholder="e.g. I've been locked out of my account since yesterday and it's very urgent...",
        height=150,
    )
    submitted = st.form_submit_button("Process ticket")

if submitted:
    if not subject.strip() or not body.strip():
        st.warning("Please fill in both the subject and the body.")
    else:
        with st.spinner("Running classification and generating response..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"subject": subject, "body": body},
                    timeout=180,
                )
                response.raise_for_status()
                result = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not reach the API. Make sure it's running: "
                    "`uvicorn api:app --reload --port 8000` in another terminal."
                )
                st.stop()
            except requests.exceptions.HTTPError as e:
                st.error(f"API returned an error: {e}")
                st.stop()

        st.success("Done!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Classification")
            preds = result.get("predictions", {})
            st.metric("Type", preds.get("type", "—"))
            st.metric("Priority", preds.get("priority", "—"))
            st.metric("Queue", preds.get("queue", "—"))

            st.subheader("🙂 Sentiment & Urgency")
            sentiment = result.get("sentiment_analysis", {})
            st.metric("Sentiment", sentiment.get("sentiment", "—"))
            st.metric("Urgency (1-5)", sentiment.get("urgency_score", "—"))
            if sentiment.get("rationale"):
                st.caption(sentiment["rationale"])

        with col2:
            st.subheader("📝 Summary")
            st.info(result.get("summary", "—"))

            st.subheader("✅ Groundedness Check")
            grounded = result.get("groundedness", {})
            is_grounded = grounded.get("is_grounded")
            if is_grounded is True:
                st.success(f"Grounded: {grounded.get('notes', '')}")
            elif is_grounded is False:
                st.error(f"Not grounded: {grounded.get('notes', '')}")
            else:
                st.warning(grounded.get("notes", "Groundedness check unavailable."))

        st.subheader("💬 Generated Response")
        st.write(result.get("generated_response", "—"))

        with st.expander("🔍 Retrieved similar tickets (raw context)"):
            contexts = result.get("retrieved_contexts", [])
            if contexts:
                for i, ctx in enumerate(contexts, start=1):
                    st.markdown(f"**{i}.** {ctx.get('answer', ctx.get('text', ctx))}")
            else:
                st.write("No contexts retrieved.")
