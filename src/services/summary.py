"""Summary generation service (updated to current LangChain style)."""

import asyncio
from datetime import datetime
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

from src.models.schemas import (
    DocumentMetadata,
    SummaryResponse,
    SummaryType,
)
from src.services.model_manager import ModelManager
from src.utils.logging import LoggerMixin, get_logger, log_execution_time


class SummaryGenerationError(Exception):
    """Raised when there's an error generating summaries."""


class SummaryGenerator(LoggerMixin):
    """Generates summaries using different LLM providers (using updated LangChain APIs)."""

    # Prompt templates for different summary types
    SUMMARY_TEMPLATES = {
        SummaryType.BRIEF: (
            "Provide a brief summary of the following text in 2-3 sentences:\n"
            "{text}\n"
            "Brief Summary:"
        ),
        SummaryType.DETAILED: (
            "Provide a detailed summary of the following text, capturing main "
            "points and key details:\n{text}\nDetailed Summary:"
        ),
        SummaryType.BULLET_POINTS: (
            "Summarize the key points of the following text in a bullet-point "
            "format:\n{text}\nKey Points:"
        ),
    }

    def __init__(
        self,
        model_manager: ModelManager,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        max_concurrent_chunks: int = 5,
    ):
        """Initialize summary generator.

        Args:
            model_manager: ModelManager instance to use.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
            max_concurrent_chunks: Maximum chunks to process concurrently.
        """
        self.logger = get_logger(self.__class__.__name__, level="DEBUG")
        self.logger.info(
            "Initializing SummaryGenerator with chunk_size=%d, chunk_overlap=%d, "
            "max_concurrent_chunks=%d",
            chunk_size,
            chunk_overlap,
            max_concurrent_chunks,
        )
        self.model_manager = model_manager
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        self.max_concurrent_chunks = max_concurrent_chunks

    def _create_prompt(
        self, summary_type: SummaryType, language: Optional[str] = None
    ) -> ChatPromptTemplate:
        """Create a ChatPromptTemplate for summary generation.

        Args:
            summary_type: Type of summary to generate.
            language: Optional target language.

        Returns:
            ChatPromptTemplate.
        """
        template = self.SUMMARY_TEMPLATES[summary_type]
        if language and language.lower() != "en":
            self.logger.debug("Adding language directive to prompt: %s", language)
            template += f"\n\nPlease provide the summary in {language}."

        messages = [
            ("system", "You are a proficient summarizer."),
            ("human", template),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        self.logger.debug("Created prompt template: %s", prompt)
        return prompt

    async def _generate_chunk_summary(
        self, chunk: str, summary_type: SummaryType, language: Optional[str] = None
    ) -> str:
        """Generate summary for a single chunk.

        Args:
            chunk: Text chunk to summarize.
            summary_type: Type of summary to generate.
            language: Optional target language.

        Returns:
            Summary of the chunk.

        Raises:
            SummaryGenerationError: If generation fails.
        """
        try:
            self.logger.debug("Generating summary for chunk (length %d)", len(chunk))
            prompt = self._create_prompt(summary_type, language)
            formatted_messages = prompt.format_messages(text=chunk)
            response = await self.model_manager.invoke(formatted_messages)
            return response.content

        except Exception as e:
            self.logger.error("Error generating chunk summary: %s", str(e))
            raise SummaryGenerationError(
                f"Failed to generate chunk summary: {str(e)}"
            ) from e

    async def _combine_summaries(
        self,
        summaries: List[str],
        summary_type: SummaryType,
        language: Optional[str] = None,
    ) -> str:
        """Combine multiple chunk summaries.

        Args:
            summaries: List of chunk summaries.
            summary_type: Type of summary.
            language: Optional target language.

        Returns:
            Combined summary.

        Raises:
            SummaryGenerationError: If combination fails.
        """
        self.logger.info("Combining %d chunk summaries.", len(summaries))
        if not summaries:
            self.logger.warning("No chunk summaries available to combine.")
            return ""

        if len(summaries) == 1:
            self.logger.debug("Only one chunk summary available; skipping combination.")
            return summaries[0]

        combined_text = "\n\n".join(summaries)
        self.logger.debug("Combined text length: %d", len(combined_text))
        template = (
            f"Below are summaries of different sections. "
            f"Combine them into a single coherent {summary_type.value} summary:\n\n"
            f"{combined_text}\n\nCombined Summary:"
        )

        if language and language.lower() != "en":
            self.logger.debug(
                "Adding language directive to combined summary prompt: %s", language
            )
            template += f"\n\nPlease provide the summary in {language}."

        try:
            messages = [
                ("system", "You are a proficient summarizer."),
                ("human", template),
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            formatted_messages = prompt.format_messages()
            response = await self.model_manager.invoke(formatted_messages)
            return response.content

        except Exception as e:
            self.logger.error("Error combining summaries: %s", str(e))
            raise SummaryGenerationError(
                f"Failed to combine summaries: {str(e)}"
            ) from e

    def _process_batch(
        self,
        chunk_summaries: List[str],
        failed_chunks: List[int],
        batch_results: List[str | Exception],
        batch_start_idx: int,
    ) -> None:
        """Process a batch of chunk summary results.

        Args:
            chunk_summaries: List to append successful summaries to.
            failed_chunks: List to append failed chunk indices to.
            batch_results: Results from the current batch.
            batch_start_idx: Starting index of the current batch.
        """
        for idx, result in enumerate(batch_results):
            chunk_idx = batch_start_idx + idx
            if isinstance(result, Exception):
                self.logger.warning("Chunk %d failed: %s", chunk_idx, result)
                failed_chunks.append(chunk_idx)
            else:
                self.logger.debug("Chunk %d summary generated successfully", chunk_idx)
                chunk_summaries.append(result)

    @log_execution_time()
    async def generate_summary(
        self,
        text: str,
        metadata: DocumentMetadata,
        summary_type: SummaryType = SummaryType.BRIEF,
        language: Optional[str] = None,
    ) -> SummaryResponse:
        """Generate a summary of the given text.

        Args:
            text: Text to summarize.
            metadata: Document metadata.
            summary_type: Type of summary to generate.
            language: Optional target language.

        Returns:
            SummaryResponse with the generated summary.

        Raises:
            SummaryGenerationError: If generation fails.
        """
        start_time = datetime.now()
        try:
            chunks = self.text_splitter.split_text(text)
            self.logger.info("Split text into %d chunks", len(chunks))

            chunk_summaries = []
            failed_chunks = []

            for i in range(0, len(chunks), self.max_concurrent_chunks):
                batch = chunks[i : i + self.max_concurrent_chunks]
                self.logger.debug(
                    "Processing batch of chunks (indexes %d to %d)",
                    i,
                    i + len(batch) - 1,
                )
                chunk_tasks = [
                    self._generate_chunk_summary(chunk, summary_type, language)
                    for chunk in batch
                ]
                batch_results = await asyncio.gather(
                    *chunk_tasks, return_exceptions=True
                )
                self._process_batch(chunk_summaries, failed_chunks, batch_results, i)

            if failed_chunks:
                self.logger.warning(
                    "Failed to process %d chunks: %s", len(failed_chunks), failed_chunks
                )

            final_summary = await self._combine_summaries(
                chunk_summaries, summary_type, language
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            token_count = len(final_summary) // 4  # Rough estimate

            self.logger.info(
                "Summary generation finished in %.3f seconds with estimated %d tokens",
                processing_time,
                token_count,
            )

            return SummaryResponse(
                summary=final_summary,
                metadata=metadata,
                provider=self.model_manager.get_current_provider(),
                summary_type=summary_type,
                processing_time=processing_time,
                token_count=token_count,
            )

        except Exception as e:
            self.logger.error("Error generating summary: %s", str(e))
            raise SummaryGenerationError(f"Failed to generate summary: {str(e)}") from e
