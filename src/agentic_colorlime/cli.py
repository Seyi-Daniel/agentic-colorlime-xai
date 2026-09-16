from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from .config_io import apply_overrides, load_profile
from .image_io import load_image_from_path


def default_profile_path() -> Path:
    return Path("configs/agent-demo.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the code-grounded agentic Color-LIME selector."
    )
    parser.add_argument("--image", required=True, nargs="+", action="extend",
                        help="One or more local image paths; --image may be repeated")
    parser.add_argument("--config", type=Path, default=default_profile_path())
    parser.add_argument("--model", help="Override the profile's Hugging Face model")
    parser.add_argument("--openai-model", default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--lime-samples", type=int, default=None)
    parser.add_argument("--lime-batch-size", type=int, default=None)
    parser.add_argument("--lime-lasso-alpha", type=float, default=None)
    parser.add_argument("--shap-samples", type=int, default=None)
    parser.add_argument("--shap-max-segments", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    parser.add_argument("--critical-area", type=float, default=None)
    parser.add_argument("--colorlime-k", type=int, default=None)
    parser.add_argument(
        "--send-visuals-to-agent",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether evidence images are sent to the agent.",
    )
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    config, profile = load_profile(args.config)
    application = profile.get("application", {})

    config = apply_overrides(
        config,
        lime_num_samples=args.lime_samples,
        lime_batch_size=args.lime_batch_size,
        lime_lasso_alpha=args.lime_lasso_alpha,
        shap_num_samples=args.shap_samples,
        shap_max_segments=args.shap_max_segments,
        inference_batch_size=args.inference_batch_size,
        critical_area_fraction=args.critical_area,
        colorlime_k=args.colorlime_k,
    )

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing. Put it in .env or export it locally.")

    model_id = args.model or application.get("model_id", "google/vit-base-patch16-224")
    openai_model = (
        args.openai_model
        or os.getenv("OPENAI_MODEL")
        or application.get("openai_model", "gpt-5-mini")
    )
    device = args.device or application.get("device", "auto")
    output_root = args.output_root or application.get("output_root", "outputs")
    send_visuals = (
        args.send_visuals_to_agent
        if args.send_visuals_to_agent is not None
        else bool(application.get("send_visuals_to_agent", True))
    )

    # Import the heavy model runtime only after arguments and credentials validate.
    from .pipeline import run_experiment

    if len(args.image) > 1:
        from .batch import ImageInput, run_batch

        batch = run_batch(
            images=[ImageInput(name=Path(path).name, source=path) for path in args.image],
            model_id_or_path=str(model_id), openai_api_key=api_key,
            openai_model=str(openai_model), hf_token=os.getenv("HF_TOKEN") or None,
            device=str(device), output_root=str(output_root),
            send_visuals_to_agent=send_visuals, config=config,
        )
        print(json.dumps(batch.summary(), indent=2))
        if batch.summary()["failed"]:
            raise SystemExit(1)
        return

    result = run_experiment(
        image=load_image_from_path(args.image[0]),
        model_id_or_path=str(model_id),
        openai_api_key=api_key,
        openai_model=str(openai_model),
        hf_token=os.getenv("HF_TOKEN") or None,
        device=str(device),
        output_root=str(output_root),
        send_visuals_to_agent=send_visuals,
        config=config,
    )
    print(
        json.dumps(
            {
                "profile": profile.get("name"),
                "prediction": result.target_class_label,
                "prediction_probability": result.target_probability,
                "selected_method": result.decision.final_method,
                "selected_candidate": result.selected_candidate,
                "tools_executed": result.decision.tools_executed,
                "tools_failed": result.decision.tools_failed,
                "tools_not_executed": result.decision.tools_not_executed,
                "run_dir": result.run_dir,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
