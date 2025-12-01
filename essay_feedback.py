import streamlit as st
import os
import operator
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

# -----------------------------
# Load environment
# -----------------------------
load_dotenv()

# -----------------------------
# Structured Schema
# -----------------------------
class EvaluateEssaySchema(BaseModel):
    feedback: str = Field(description="Short feedback for the essay section")
    score: int = Field(description="Score out of 10", ge=0, le=10)

# -----------------------------
# LLM Model (OpenRouter)
# -----------------------------
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("hf"),
    model="mistralai/mistral-7b-instruct:free",
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Essay Evaluator App"
    }
)

structured = model.with_structured_output(EvaluateEssaySchema)

# -----------------------------
# LangGraph State
# -----------------------------
class EssayState(TypedDict):
    user_input: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    summary_feedback: str
    overall_score: Annotated[list[int], operator.add]
    avg_score: float

# -----------------------------
# Evaluation Node Functions
# -----------------------------
def language_evaluation(state: EssayState):
    prompt = f"Evaluate the LANGUAGE quality of this essay. Give feedback + score out of 10:\n\n{state['user_input']}"
    result = structured.invoke(prompt)
    return {"language_feedback": result.feedback, "overall_score": [result.score]}

def analysis_evaluation(state: EssayState):
    prompt = f"Evaluate the ANALYSIS quality of this essay. Give feedback + score out of 10:\n\n{state['user_input']}"
    result = structured.invoke(prompt)
    return {"analysis_feedback": result.feedback, "overall_score": [result.score]}

def clarity_evaluation(state: EssayState):
    prompt = f"Evaluate the CLARITY of this essay. Give feedback + score out of 10:\n\n{state['user_input']}"
    result = structured.invoke(prompt)
    return {"clarity_feedback": result.feedback, "overall_score": [result.score]}

def summary_evaluation(state: EssayState):
    avg_score = sum(state["overall_score"]) / len(state["overall_score"])

    prompt = f"""
Summarize the feedback for this essay based on:

Language: {state["language_feedback"]}
Analysis: {state["analysis_feedback"]}
Clarity: {state["clarity_feedback"]}

Also validate this average score: {avg_score}
And rewrite the feedback in a clean, concise style.
    """

    result = structured.invoke(prompt)
    return {"summary_feedback": result.feedback, "avg_score": avg_score}

# -----------------------------
# Build LangGraph Workflow
# -----------------------------
graph = StateGraph(EssayState)

graph.add_node("language_evaluation", language_evaluation)
graph.add_node("analysis_evaluation", analysis_evaluation)
graph.add_node("clarity_evaluation", clarity_evaluation)
graph.add_node("summary_evaluation", summary_evaluation)

graph.add_edge(START, "language_evaluation")
graph.add_edge(START, "analysis_evaluation")
graph.add_edge(START, "clarity_evaluation")

graph.add_edge("language_evaluation", "summary_evaluation")
graph.add_edge("analysis_evaluation", "summary_evaluation")
graph.add_edge("clarity_evaluation", "summary_evaluation")

graph.add_edge("summary_evaluation", END)

workflow = graph.compile()

# -------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------
st.set_page_config(
    page_title="Essay Evaluation AI",
    page_icon="📝",
    layout="centered"
)

st.title("📝 AI Essay Evaluation System")
st.write("Evaluate your essay's **Language**, **Analysis**, **Clarity**, and get a **Final Summary Score**.")

st.markdown("---")

essay_text = st.text_area("✍️ Paste Your Essay Below:", height=250)

if st.button("🚀 Evaluate Essay"):
    if essay_text.strip() == "":
        st.warning("Please enter an essay first.")
    else:
        with st.spinner("Analyzing your essay... ⏳"):
            result = workflow.invoke({"user_input": essay_text})

        st.success("Evaluation Complete!")

        # ----------------------------
        # SECTION SCORES + FEEDBACK
        # ----------------------------
        st.subheader("📌 Language Feedback")
        st.info(result["language_feedback"])

        st.subheader("📌 Analysis Feedback")
        st.info(result["analysis_feedback"])

        st.subheader("📌 Clarity Feedback")
        st.info(result["clarity_feedback"])

        # ----------------------------
        # FINAL SUMMARY
        # ----------------------------
        st.subheader("🏁 Final Summary Feedback")
        st.success(result["summary_feedback"])

        # ----------------------------
        # SCORES
        # ----------------------------
        st.markdown("### 📊 Scores")
        st.write(f"**Language Score:** {result['overall_score'][0]}")
        st.write(f"**Analysis Score:** {result['overall_score'][1]}")
        st.write(f"**Clarity Score:** {result['overall_score'][2]}")
        st.write(f"### 🧮 **Average Score:** `{result['avg_score']:.2f}` / 10")

st.markdown("---")
st.write("⚡ Powered by LangGraph + OpenRouter + Streamlit")
