from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agentic_colorlime.config import ExperimentConfig
from agentic_colorlime.image_io import load_image_from_path, load_image_from_upload
from agentic_colorlime.model_runner import HuggingFaceImageClassifier
from agentic_colorlime.pipeline import run_experiment
from agentic_colorlime.segmentations import SEGMENTATION_FUNCTIONS

load_dotenv()

st.set_page_config(page_title="Agentic Color-LIME XAI", layout="wide")
st.title("Agentic Color-LIME XAI")
st.caption(
    "The agent can inspect local tool code, executes one method at a time, "
    "uses CIR as evidence, reviews counterevidence, and decides when to stop. "
    "Parameters shown below are human-configured in this research version."
)

with st.sidebar:
    st.header("Inputs")
    source = st.radio("Image source", ["Upload", "Local path"], horizontal=True)
    uploaded = None
    local_path = ""
    if source == "Upload":
        uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    else:
        local_path = st.text_input("Image path on the machine running Streamlit")

    model_id = st.text_input(
        "Hugging Face model ID or local model directory",
        "google/vit-base-patch16-224",
    )
    openai_model = st.text_input("OpenAI agent model", os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    api_key = st.text_input(
        "OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Prefer storing OPENAI_API_KEY in .env. It is never written to run outputs.",
    )
    hf_token = st.text_input(
        "Hugging Face token (gated/private models only)",
        os.getenv("HF_TOKEN", ""),
        type="password",
    )
    device = st.selectbox("Device", ["auto", "cuda", "mps", "cpu"])
    send_visuals = st.checkbox(
        "Let the agent see the original and generated evidence images",
        value=True,
        help=(
            "Recommended. The agent receives downscaled copies of the original, "
            "segmentation boundaries, LIME overlay, and CIR omission image. When "
            "disabled, it is forbidden from making visual claims."
        ),
    )

    st.header("Agent policy")
    st.info(
        f"All {len(SEGMENTATION_FUNCTIONS)} registered explanation tools are exposed. "
        "There is no user-set minimum or maximum. The agent may inspect local "
        "function source, but no hand-written strengths or limitations are supplied. "
        "After every CIR calculation, a separate evidence review is required before "
        "the agent may continue or finish."
    )

    st.header("LIME and CIR")
    lime_samples = st.number_input("LIME samples per attempted method", 50, 10000, 500, step=50)
    lime_batch = st.number_input("LIME outer batch size", 1, 512, 32)
    inference_batch = st.number_input("ViT inference microbatch", 1, 256, 16)
    critical_area = st.slider("Critical area fraction", 0.05, 0.50, 0.20, 0.01)
    color_k = st.number_input("ColorLIME K", 2, 2048, 128)

    with st.expander("Segmentation parameters"):
        slic_segments = st.number_input("SLIC target segments", 10, 1000, 100)
        watershed_markers = st.number_input("Watershed markers", 10, 1000, 100)
        f_scale = st.number_input("Felzenszwalb scale", 1.0, 1000.0, 100.0)
        q_kernel = st.number_input("Quickshift kernel size", 1, 20, 3)
        q_max_dist = st.number_input("Quickshift max distance", 1, 50, 6)

    run_clicked = st.button("Run explanation agent", type="primary", use_container_width=True)

if source == "Upload" and uploaded is not None:
    image_array = load_image_from_upload(uploaded.getvalue())
    st.image(image_array, caption="Input image", width=450)
elif source == "Local path" and local_path.strip():
    try:
        image_array = load_image_from_path(local_path)
        st.image(image_array, caption="Input image", width=450)
    except Exception as exc:
        image_array = None
        st.error(str(exc))
else:
    image_array = None


@st.cache_resource(show_spinner="Loading Hugging Face model…")
def load_predictor(model_id_or_path: str, device_name: str, batch_size: int, token: str):
    return HuggingFaceImageClassifier(
        model_id_or_path,
        device=device_name,
        inference_batch_size=batch_size,
        hf_token=token or None,
    )


if run_clicked:
    if image_array is None:
        st.error("Provide an image first.")
        st.stop()
    if not model_id.strip():
        st.error("Provide a Hugging Face model ID or local directory.")
        st.stop()
    if not api_key.strip():
        st.error("Provide OPENAI_API_KEY in .env or the sidebar.")
        st.stop()

    config = ExperimentConfig(
        lime_num_samples=int(lime_samples),
        lime_batch_size=int(lime_batch),
        inference_batch_size=int(inference_batch),
        critical_area_fraction=float(critical_area),
        colorlime_k=int(color_k),
        slic_n_segments=int(slic_segments),
        watershed_markers=int(watershed_markers),
        felzenszwalb_scale=float(f_scale),
        quickshift_kernel_size=int(q_kernel),
        quickshift_max_dist=int(q_max_dist),
    )

    predictor = load_predictor(model_id.strip(), device, int(inference_batch), hf_token.strip())
    with st.status("The agent is deciding which explanation tool to use…", expanded=True) as status:
        st.write(
            f"All {len(SEGMENTATION_FUNCTIONS)} registered methods are available to the agent. "
            "It may inspect code, run one or more methods, and must review the "
            "evidence before explaining why it stops."
        )
        st.write("The first model run may download and cache Hugging Face files.")
        result = run_experiment(
            image=image_array,
            model_id_or_path=model_id.strip(),
            openai_api_key=api_key.strip(),
            openai_model=openai_model.strip(),
            hf_token=hf_token.strip() or None,
            device=device,
            send_visuals_to_agent=send_visuals,
            config=config,
            predictor=predictor,
        )
        status.update(label="Agent selection completed", state="complete", expanded=False)

    st.session_state["latest_result"] = result

result = st.session_state.get("latest_result")
if result is not None:
    st.header("Final result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted class", result.target_class_label)
    c2.metric("Prediction probability", f"{result.target_probability:.4f}")
    c3.metric("Selected segmentation", result.decision.final_method)
    c4.metric("Explanation tools attempted", result.decision.explanation_calls_used)

    selected = result.selected_candidate
    left, right = st.columns([1.25, 1])
    with left:
        st.image(
            selected["artifacts"]["explanation_overlay"],
            caption=f"Selected explanation: {result.decision.final_method}",
            use_container_width=True,
        )
    with right:
        st.metric("Selected CIR", f"{selected['cir']:.6f}")
        st.metric("Relative CIR", f"{selected['relative_cir']:.4f}")
        st.metric("Critical area used", f"{selected['critical_area_fraction']:.3f}")
        st.metric("LIME surrogate score", f"{selected['lime_local_surrogate_score']:.4f}")

    st.subheader("Agent decision")
    decision = result.decision.raw
    st.write(f"**Selection rationale:** {decision.get('rationale', 'not reported')}")
    st.write(f"**Stopping reason:** {decision.get('stopping_reason', 'not reported')}")
    st.write(f"**CIR assessment:** {decision.get('cir_assessment', 'not reported')}")
    if decision.get("why_not_highest_cir"):
        st.info(f"Why a higher-CIR candidate was not selected: {decision['why_not_highest_cir']}")

    comparison = result.decision.evidence_comparison
    st.subheader("CIR corroboration")
    e1, e2, e3 = st.columns(3)
    e1.metric("Selected CIR rank", f"{comparison['selected_cir_rank']}/{comparison['evaluated_candidate_count']}")
    e2.metric("Highest observed CIR", f"{comparison['highest_observed_cir']:.6f}")
    e3.metric("Gap from highest", f"{comparison['absolute_gap_from_highest']:.6f}")
    if comparison["selected_is_highest_observed_cir"]:
        st.success("The selected explanation also had the highest CIR among the methods actually attempted.")
    else:
        st.warning("The agent selected a lower-CIR explanation; review its stated trade-off above.")

    st.subheader("Tool execution")
    st.write(f"**Attempted:** {', '.join(result.decision.tools_executed) or 'none'}")
    if result.decision.tools_failed:
        st.write("**Failed attempts:**")
        st.json(result.decision.tools_failed)
    st.write(f"**Not executed:** {', '.join(result.decision.tools_not_executed) or 'none'}")
    inspected = decision.get("inspected_source_methods", [])
    st.write(f"**Source inspected:** {', '.join(inspected) or 'none'}")

    reviews = decision.get("evidence_reviews", [])
    if reviews:
        st.subheader("Evidence reviews")
        for review in reviews:
            label = (
                f"{review.get('method', 'candidate')} → "
                f"{review.get('next_action', 'unknown')}"
            )
            with st.expander(label):
                st.write(
                    f"**Evidence summary:** "
                    f"{review.get('evidence_summary', 'not reported')}"
                )
                st.write(
                    f"**Counterevidence considered:** "
                    f"{review.get('counterevidence_considered', 'not reported')}"
                )
                st.write(
                    f"**Action reason:** "
                    f"{review.get('action_reason', 'not reported')}"
                )

    frame = pd.DataFrame(result.candidates)
    visible_columns = [
        "method",
        "selection_rationale",
        "pre_call_confidence",
        "number_of_segments",
        "cir",
        "relative_cir",
        "critical_area_fraction",
        "lime_local_surrogate_score",
        "total_seconds",
        "decision_changed",
    ]
    st.dataframe(
        frame[[column for column in visible_columns if column in frame.columns]],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Attempted candidate explanations")
    columns = st.columns(min(3, max(1, len(result.candidates))))
    for index, candidate in enumerate(result.candidates):
        with columns[index % len(columns)]:
            st.image(
                candidate["artifacts"]["explanation_overlay"],
                caption=f"{candidate['method']} | CIR={candidate.get('cir', float('nan')):.4f}",
                use_container_width=True,
            )
            st.caption(candidate.get("selection_rationale", ""))

    suggestion = decision.get("alternative_xai_suggestion", "none")
    reason = decision.get("alternative_xai_reason", "")
    st.subheader("Optional alternative XAI suggestion")
    st.write(f"**{suggestion}**" + (f" — {reason}" if reason else ""))

    st.subheader("Decision timeline")
    timeline_rows = []
    for index, event in enumerate(result.decision.trace, start=1):
        row = {
            "step": index,
            "event": event.get("event"),
            "tool": event.get("tool"),
            "method": event.get("method")
            or event.get("result", {}).get("method")
            or event.get("arguments", {}).get("method"),
        }
        if event.get("event") == "evidence_review":
            row["decision"] = event.get("arguments", {}).get("next_action")
            row["reason"] = event.get("arguments", {}).get("action_reason")
        elif event.get("event") == "final_selection":
            row["decision"] = "finish"
            row["reason"] = event.get("result", {}).get("stopping_reason")
        elif event.get("event") == "explanation_tool":
            row["decision"] = "execute"
            row["reason"] = event.get("arguments", {}).get("selection_rationale")
        elif event.get("event") == "source_inspection":
            row["decision"] = "inspect source"
        elif event.get("event") == "cir_tool":
            row["decision"] = "calculate CIR"
            row["cir"] = event.get("result", {}).get("cir")
        timeline_rows.append(row)
    st.dataframe(pd.DataFrame(timeline_rows), use_container_width=True, hide_index=True)

    with st.expander("Complete agent tool trace"):
        st.json(result.decision.trace)
    with st.expander("Cheap image profile used before tool selection"):
        st.json(result.image_profile)
    st.caption(f"Run outputs: {result.run_dir}")
    st.caption(f"Exact LLM request/response audit: {result.decision.audit_directory}")
