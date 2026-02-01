"""Community report generation component for ManasRAG.

This component uses LLMs to generate summary reports for each detected community,
providing high-level overviews of the entities and relationships within.
"""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from haystack import component
from haystack.dataclasses import ChatMessage

from manasrag._logging import get_logger
from manasrag.core.community import Community
from manasrag.stores.base import GraphDocumentStore


@component
class CommunityReportGenerator:
    """Generate summary reports for communities.

    Uses an LLM to generate structured reports that summarize:
    - The main theme/topic of the community
    - Key entities and their roles
    - Important relationships within the community
    """

    def __init__(
        self,
        generator: Any | None = None,
        max_tokens: int = 2000,
        max_workers: int = 4,
    ):
        """Initialize the CommunityReportGenerator.

        Args:
            generator: Haystack ChatGenerator for LLM calls.
            max_tokens: Maximum tokens for community descriptions in prompts.
            max_workers: Maximum number of parallel workers for report generation.
        """
        self.generator = generator
        self.max_tokens = max_tokens
        self.max_workers = max_workers

        # Logger
        self._logger = get_logger("report_generator")

    @component.output_types(reports=dict)
    def run(
        self,
        graph_store: GraphDocumentStore,
        communities: dict[str, Community],
        max_workers: int | None = None,
    ) -> dict:
        """Generate reports for all communities.

        Args:
            graph_store: The graph store for fetching entity/relation details.
            communities: Dictionary of communities to generate reports for.
            max_workers: Override for number of parallel workers.

        Returns:
            Dictionary with:
                - reports: Dict mapping community IDs to report strings
        """
        if not communities:
            return {"reports": {}}

        if self.generator is None:
            raise ValueError("Generator not configured. Provide a ChatGenerator.")

        self._logger.info(f"Generating reports for {len(communities)} communities")

        # Process communities by level (bottom-up)
        levels = sorted(set(c.level for c in communities.values()))

        reports = {}

        for level in levels:
            level_communities = {
                k: v for k, v in communities.items() if v.level == level
            }
            self._logger.debug(f"Processing level {level}: {len(level_communities)} communities")

            # Use provided max_workers or fall back to instance setting
            workers = max_workers if max_workers is not None else self.max_workers

            if workers <= 1:
                # Sequential processing for single worker or disabled parallelism
                for comm_id, community in level_communities.items():
                    report = self._generate_single_report(
                        graph_store, community, reports
                    )
                    community.report_string = report
                    reports[comm_id] = report
            else:
                # Parallel report generation
                reports = self._generate_reports_parallel(
                    graph_store, level_communities, reports, workers
                )

        self._logger.debug(f"Generated {len(reports)} reports")

        return {"reports": reports}

    def _generate_reports_parallel(
        self,
        graph_store: GraphDocumentStore,
        level_communities: dict[str, Community],
        existing_reports: dict[str, str],
        max_workers: int,
    ) -> dict:
        """Generate reports for multiple communities in parallel.

        Args:
            graph_store: The graph store for fetching entity/relation details.
            level_communities: Dictionary of communities at this level to generate reports for.
            existing_reports: Reports from sub-communities (for hierarchical mode).
            max_workers: Maximum number of parallel workers.

        Returns:
            Updated dictionary with reports for these communities.
        """
        reports = {}
        pending_reports = existing_reports.copy()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all report generation tasks
            futures = {
                comm_id: executor.submit(
                    self._generate_report_task,
                    comm_id,
                    community,
                    graph_store,
                    pending_reports,
                )
                for comm_id, community in level_communities.items()
            }

            # Collect results as they complete
            for future in as_completed(futures.values()):
                comm_id, report = future.result()
                pending_reports[comm_id] = report
                reports[comm_id] = report

        return reports

    def _generate_report_task(
        self,
        comm_id: str,
        community: Community,
        graph_store: GraphDocumentStore,
        existing_reports: dict[str, str],
    ) -> tuple[str, str]:
        """Task wrapper for parallel report generation.

        Args:
            comm_id: Community ID.
            community: The community to generate a report for.
            graph_store: The graph store for fetching details.
            existing_reports: Reports from sub-communities.

        Returns:
            Tuple of (community_id, report_string).
        """
        report = self._generate_single_report(
            graph_store, community, existing_reports
        )
        community.report_string = report
        return comm_id, report

    def _generate_single_report(
        self,
        graph_store: GraphDocumentStore,
        community: Community,
        existing_reports: dict[str, str],
    ) -> str:
        """Generate a report for a single community.

        Args:
            graph_store: The graph store for fetching details.
            community: The community to generate a report for.
            existing_reports: Reports from sub-communities (for hierarchical mode).

        Returns:
            Generated report string.
        """
        # Build community description
        description = self._build_community_description(
            graph_store, community, existing_reports
        )

        # Generate report using LLM
        prompt = self._get_report_prompt().format(
            community_description=description
        )

        result = self._call_llm(prompt)

        # Parse and format the result
        return self._format_report(result, community)

    def _build_community_description(
        self,
        graph_store: GraphDocumentStore,
        community: Community,
        existing_reports: dict[str, str],
    ) -> str:
        """Build a description of the community for the LLM.

        Includes entities, relationships, and sub-community reports.
        """
        lines = ["## Community Description", f"**Title:** {community.title}"]

        # Add sub-community reports if available
        if community.sub_communities:
            lines.append("\n### Sub-Communities")
            for sub_id in community.sub_communities:
                if sub_id in existing_reports:
                    lines.append(f"\n{existing_reports[sub_id]}")

        # Add entities
        lines.append("\n### Entities")
        lines.append("| ID | Entity | Type | Description | Degree |")
        lines.append("|---|--------|------|-------------|--------|")

        for i, node_id in enumerate(community.nodes[:50]):  # Limit for context
            node_data = graph_store.get_node(node_id)
            if node_data:
                degree = graph_store.node_degree(node_id)
                lines.append(
                    f"| {i} | {node_id} | {node_data.get('entity_type', 'UNKNOWN')} | "
                    f"{node_data.get('description', '')[:100]} | {degree} |"
                )

        # Add relationships
        lines.append("\n### Relationships")
        lines.append("| ID | Source | Target | Description | Rank |")
        lines.append("|---|--------|--------|-------------|------|")

        for i, (src, tgt) in enumerate(community.edges[:50]):  # Limit for context
            edge_data = graph_store.get_edge(src, tgt)
            if edge_data:
                rank = graph_store.edge_degree(src, tgt)
                lines.append(
                    f"| {i} | {src} | {tgt} | "
                    f"{edge_data.get('description', '')[:100]} | {rank} |"
                )

        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        if self.generator is None:
            raise ValueError("Generator not configured.")

        self._logger.debug(f"Calling LLM (prompt_len={len(prompt)})")
        response = self.generator.run(messages=[ChatMessage.from_user(prompt)])

        # Extract text from response - handle various Haystack response formats
        if isinstance(response, dict):
            if "replies" in response and response["replies"]:
                reply = response["replies"][0]
                if isinstance(reply, dict) and "content" in reply:
                    result = reply["content"]
                elif isinstance(reply, str):
                    result = reply
                else:
                    result = str(reply)
            elif "content" in response:
                result = response["content"]
            else:
                result = str(response)
        elif hasattr(response, "replies") and response.replies:
            reply = response.replies[0]
            if hasattr(reply, "content"):
                content = reply.content
                # Handle ChatMessage with TextContent objects
                if isinstance(content, list) and content:
                    result = "".join(
                        getattr(c, "text", str(c)) for c in content
                    )
                elif isinstance(content, str):
                    result = content
                else:
                    result = str(content)
            elif hasattr(reply, "text"):
                result = reply.text
            elif isinstance(reply, dict) and "content" in reply:
                result = reply["content"]
            else:
                result = str(reply)
        else:
            result = str(response)

        self._logger.debug(f"LLM response (len={len(result)})")
        return result

    def _get_report_prompt(self) -> str:
        """Get the community report generation prompt."""
        return """You are analyzing a community of related entities in a knowledge graph.

Based on the following information, generate a concise report that:
1. Identifies the main theme/topic of this community
2. Summarizes the key entities and their roles
3. Describes important relationships

{community_description}

Generate a JSON response with this structure:
{{
    "title": "Brief descriptive title",
    "summary": "2-3 sentence overview",
    "findings": [
        {{"summary": "Key point 1", "explanation": "Details"}},
        {{"summary": "Key point 2", "explanation": "Details"}}
    ]
}}

Response:"""

    def _format_report(self, result: str, community: Community) -> str:
        """Format the LLM result into a report string.

        Also attempts to parse JSON for the report_json field.
        """
        # Try to parse JSON
        try:
            # Extract JSON from the response
            start = result.find("{")
            end = result.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                report_json = json.loads(json_str)
                community.report_json = report_json

                # Format from JSON
                title = report_json.get("title", community.title)
                summary = report_json.get("summary", "")
                findings = report_json.get("findings", [])

                sections = [f"# {title}", f"\n{summary}\n"]
                for finding in findings:
                    if isinstance(finding, dict):
                        summary_text = finding.get("summary", "")
                        explanation = finding.get("explanation", "")
                        sections.append(f"## {summary_text}\n\n{explanation}")
                    else:
                        sections.append(f"## {finding}")

                return "\n\n".join(sections)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return raw result
        return result


@component
class CommunityReportFormatter:
    """Format community reports for use in retrieval.

    Converts community reports into a format suitable for
    inclusion in retrieval context.
    """

    @component.output_types(formatted_reports=str)
    def run(
        self,
        reports: dict[str, str],
        communities: dict[str, Community],
        limit: int = 5,
    ) -> dict:
        """Format community reports for retrieval context.

        Args:
            reports: Dictionary of community report strings.
            communities: Dictionary of Community objects.
            limit: Maximum number of reports to include.

        Returns:
            Dictionary with formatted_reports string.
        """
        # Sort by occurrence (frequency)
        sorted_communities = sorted(
            communities.items(),
            key=lambda x: x[1].occurrence,
            reverse=True,
        )[:limit]

        lines = ["-----Community Reports-----"]

        for i, (comm_id, community) in enumerate(sorted_communities):
            report = reports.get(comm_id, "")
            lines.append(f"\n### Report {i}: {community.title}")
            lines.append(report[:500])  # Truncate for context

        return {"formatted_reports": "\n".join(lines)}
