"""
PII Handler for AI Refinery SDK.

This module provides functionality for detecting, masking, and demasking
personally identifiable information (PII) in text using Microsoft Presidio.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_anonymizer import AnonymizerEngine, DeanonymizeEngine
from pydantic import BaseModel


class PIIMetadata(BaseModel):
    """Model for PII masking metadata."""

    mapping_key: str
    original_text: str
    entity_type: str
    operator: str
    start: int
    end: int
    placeholder: str


class PIIHandler:
    """
    A class for handling PII detection, masking, and demasking using Microsoft Presidio.
    """

    def __init__(
        self, language: str = "en", config_path: str = "pii_handler.yaml"
    ) -> None:
        """
        Initialize the PII handler.

        Args:
            language: Language to use for PII detection. Defaults to ``"en"``.
        """
        self.logger = logging.getLogger(__name__)
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()
        self.deanonymizer = DeanonymizeEngine()

        self.language = language
        self.pii_mapping: Dict[str, Dict[str, Any]] = {}
        self.enabled = False
        self.entity_counts: Dict[str, int] = {}
        self.metadata_store: List[PIIMetadata] = []

        self.config_path = config_path
        self._load_config(config_path)

    # Control helpers
    def enable(self) -> None:  # pragma: no cover
        """Enable PII protection."""
        self.enabled = True

    def disable(self) -> None:  # pragma: no cover
        """Disable PII protection."""
        self.enabled = False

    def is_enabled(self) -> bool:
        """Return ``True`` if protection is enabled."""
        return self.enabled

    def clear_mapping(self) -> None:
        """Clear the internal mapping used for deanonymization."""
        self.pii_mapping.clear()
        self.entity_counts.clear()

    def get_metadata(self) -> List[PIIMetadata]:
        """Get current PII metadata."""
        return self.metadata_store

    def extend_metadata(self, new_entries: List[PIIMetadata]) -> None:
        """Extend internal metadata store with new entries."""
        existing_keys = {m.mapping_key for m in self.metadata_store}
        filtered_new = [m for m in new_entries if m.mapping_key not in existing_keys]
        self.metadata_store.extend(filtered_new)

    def clear_metadata(self) -> None:
        """Clear stored metadata."""
        self.metadata_store = []

    # Core: mask / demask
    def mask_text(self, text: str) -> Tuple[str, List[PIIMetadata]]:
        """
        Detect and mask PII entities in a given text string using configurable placeholder formats.

        This method identifies PII entities using Presidio, replaces each entity with a
        consistent placeholder (e.g., [EMAIL_1]), and returns the masked string along with
        metadata necessary for reversible demasking.

        Args:
            text (str): The input text containing potential PII.

        Returns:
            Tuple[str, List[PIIMetadata]]:
                - The text with PII entities replaced by placeholders.
                - A list of PIIMetadata objects for each detected entity, containing
                mapping_key, original_text, entity_type, placeholder, etc.
        """
        if not self.enabled or not text:
            return text, []

        analyzer_results = self.analyzer.analyze(
            text=text,
            entities=self.COMMON_ENTITIES,
            language=self.language,
        )
        analyzer_results = deduplicate_overlapping_results(analyzer_results)
        if not analyzer_results:
            self.logger.info("No PII detected in the query.")
            return text, []

        metadata: List[PIIMetadata] = []
        seen_placeholders: Dict[Tuple[str, str], str] = {}

        # Preloading from existing metadata
        for m in self.metadata_store:
            seen_placeholders[(m.entity_type, m.original_text)] = m.placeholder
            entity_num = int(re.findall(r"_(\d+)]?$", m.placeholder)[0])
            self.entity_counts[m.entity_type] = max(
                self.entity_counts.get(m.entity_type, 0), entity_num
            )

        # Build operators dict
        for r in analyzer_results:
            original_text = text[r.start : r.end]
            entity_type = r.entity_type
            reuse_key = (entity_type, original_text)

            if reuse_key in seen_placeholders:
                placeholder = seen_placeholders[reuse_key]
            else:
                self.entity_counts[entity_type] = (
                    self.entity_counts.get(entity_type, 0) + 1
                )
                placeholder = f"[{entity_type}_{self.entity_counts[entity_type]}]"
                seen_placeholders[reuse_key] = placeholder

            # Add to metadata
            mapping_key = f"{entity_type}_{r.start}_{r.end}"
            metadata.append(
                PIIMetadata(
                    mapping_key=mapping_key,
                    original_text=original_text,
                    entity_type=entity_type,
                    operator=self.ENTITY_OPERATOR_MAPPING.get(
                        entity_type, ("replace", {})
                    )[0],
                    start=r.start,
                    end=r.end,
                    placeholder=placeholder,
                )
            )

        # Replace PII manually in text
        masked_text = text
        for m in sorted(
            metadata, key=lambda x: -x.start
        ):  # replace from end to avoid shifting offsets
            masked_text = masked_text[: m.start] + m.placeholder + masked_text[m.end :]

        self.logger.info("Detected and masked the following PII types:")
        print("\n[PII MASKING] Detected and masked the following PII types:")
        for item in metadata:
            self.logger.info(
                f" - {item.entity_type} at [{item.start}:{item.end}] "
                f"-> '{item.original_text}' -> {item.placeholder}"
            )
            print(
                f" - {item.entity_type} at [{item.start}:{item.end}] "
                f"-> '{item.original_text}' -> {item.placeholder}"
            )

        return masked_text, metadata

    def demask_text(
        self,
        text: str,
        metadata: Optional[List[PIIMetadata]] = None,
    ) -> str:
        """
        Restore the original PII values in a text string based on masking metadata.

        This method replaces each placeholder (e.g., [EMAIL_1]) in the text with the
        corresponding original value as recorded in the metadata.

        Args:
            text (str): The input text containing placeholders.
            metadata (Optional[List[PIIMetadata]]): The list of PIIMetadata objects
                produced during a prior call to `mask_text`.

        Returns:
            str: The text with all placeholders replaced by the original PII values.
        """
        if not self.enabled or not text or not metadata:
            return text

        for m in metadata:
            placeholder = m.placeholder
            if placeholder in text:
                text = text.replace(placeholder, m.original_text)

        # Optional cleanup
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\s+([,.!?])", r"\1", text)
        return text.strip()

    # JSON helpers
    def mask_json(
        self,
        json_data: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[PIIMetadata]]:
        """Recursively mask all string values in a JSON-serialisable object."""
        if not self.enabled:
            return json_data, []

        all_meta: List[PIIMetadata] = []

        def _process(item: Any) -> Any:
            if isinstance(item, str):
                masked, meta = self.mask_text(item)
                all_meta.extend(meta)
                return masked
            if isinstance(item, dict):
                return {k: _process(v) for k, v in item.items()}
            if isinstance(item, list):
                return [_process(elem) for elem in item]
            return item

        return _process(json_data), all_meta

    def demask_json(
        self,
        json_data: Dict[str, Any],
        metadata: List[PIIMetadata],
    ) -> Dict[str, Any]:
        """Recursively restore PII inside a JSON-serialisable object."""
        if not self.enabled or not metadata:
            return json_data

        def _process(item: Any) -> Any:
            if isinstance(item, str):
                return self.demask_text(item, metadata)
            if isinstance(item, dict):
                return {k: _process(v) for k, v in item.items()}
            if isinstance(item, list):
                return [_process(elem) for elem in item]
            return item

        return _process(json_data)

    def _load_config(self, config_path: str) -> None:
        # Load YAML config using OmegaConf
        resolved_path = Path(config_path)
        if not resolved_path.is_absolute():
            resolved_path = Path(__file__).parent / resolved_path
        config = OmegaConf.load(str(resolved_path))

        # Convert to dict to avoid typing issues with OmegaConf
        config_dict = OmegaConf.to_container(config)
        if not isinstance(config_dict, dict):
            config_dict = {}

        # Use the dictionary directly
        self.COMMON_ENTITIES = config_dict.get("common_entities", [])
        entity_mapping = config_dict.get("entity_operator_mapping", {})
        self.ENTITY_OPERATOR_MAPPING = {
            entity: (entry.get("operator"), dict(entry.get("params") or {}))
            for entity, entry in entity_mapping.items()
        }

    def load_runtime_overrides(self, config_dict: Optional[Any]) -> None:
        """
        Load entity/operator overrides from distiller YAML.

        Assumes PIIHandler is already instantiated and enabled.
        """
        if not config_dict:
            self.logger.info("No distiller config provided. Using defaults.")
            self._load_config(self.config_path)
            return

        try:
            if not isinstance(config_dict, dict):
                self.logger.warning("Invalid config format, expected a dictionary.")
                config_dict = {}
            pii_cfg = config_dict.get("base_config", {}).get("pii_masking", {})

            custom_config = pii_cfg.get("config", {})
            if not custom_config:
                self.logger.info("No overrides provided, using pii_handler.yaml.")
                self._load_config(self.config_path)
                return

            self.COMMON_ENTITIES = custom_config.get(
                "common_entities", self.COMMON_ENTITIES
            )

            mapping = custom_config.get("entity_operator_mapping", {})
            self.ENTITY_OPERATOR_MAPPING = {
                entity: (
                    entry.get("operator", "replace"),
                    dict(entry.get("params") or {}),
                )
                for entity, entry in mapping.items()
            }

            self.logger.info("PII overrides loaded from distiller YAML.")

        except Exception as e:
            self.logger.error(f"Failed to load overrides from YAML: {e}")
            self.logger.info("Falling back to pii_handler.yaml.")
            self._load_config(self.config_path)


def deduplicate_overlapping_results(
    results: List[RecognizerResult],
) -> List[RecognizerResult]:
    """
    Remove overlapping recognizer results by keeping the highest scoring result
    for each overlapping region.
    """
    results = sorted(results, key=lambda r: (r.start, -r.score))
    final = []
    for r in results:
        if not any(
            existing.start < r.end and r.start < existing.end for existing in final
        ):
            final.append(r)
    return final
