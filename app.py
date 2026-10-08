import asyncio

import chainlit as cl

from src.pipelines.advanced_rag import AdvancedRAGPipeline
from src.pipelines.naive_rag import NaiveRAGPipeline

PIPELINE_NAIVE = "Naive RAG"
PIPELINE_ADVANCED = "Advanced RAG"
DEFAULT_PIPELINE = PIPELINE_NAIVE

PIPELINE_FACTORIES = {
    PIPELINE_NAIVE: NaiveRAGPipeline,
    PIPELINE_ADVANCED: AdvancedRAGPipeline,
}


def create_pipeline(pipeline_name: str):
    factory = PIPELINE_FACTORIES.get(pipeline_name, NaiveRAGPipeline)
    return factory()


def _set_session_pipeline(pipeline_name: str) -> None:
    cl.user_session.set("pipeline_name", pipeline_name)
    cl.user_session.set("rag_pipeline", create_pipeline(pipeline_name))


@cl.on_chat_start
async def start():
    _set_session_pipeline(DEFAULT_PIPELINE)

    await cl.ChatSettings(
        [
            cl.input_widget.Select(
                id="pipeline",
                label="RAG Pipeline",
                values=[PIPELINE_NAIVE, PIPELINE_ADVANCED],
                initial_index=0,
            )
        ]
    ).send()

    await cl.Message(
        content=(
            "Ask AMAN HR is ready.\n\n"
            f"Current pipeline: **{DEFAULT_PIPELINE}**\n"
            "Use the settings panel to switch between Naive RAG and Advanced RAG."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings):
    pipeline_name = settings.get("pipeline", DEFAULT_PIPELINE)
    _set_session_pipeline(pipeline_name)
    await cl.Message(content=f"Switched to **{pipeline_name}**.").send()


@cl.on_message
async def main(message: cl.Message):
    rag_pipeline = cl.user_session.get("rag_pipeline")
    pipeline_name = cl.user_session.get("pipeline_name", DEFAULT_PIPELINE)

    res = await asyncio.to_thread(rag_pipeline.query, message.content)

    if res and "answer" in res:
        answer = res["answer"]
    else:
        answer = "Error: Failed to process query."

    await cl.Message(content=f"**[{pipeline_name}]**\n\n{answer}").send()
