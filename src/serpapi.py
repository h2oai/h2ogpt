import functools
import os
import sys
import typing
from typing import Any, Dict, Optional, Tuple

import aiohttp
from langchain.docstore.document import Document
from langchain.pydantic_v1 import Extra, Field, root_validator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utilities.serpapi import HiddenPrints
from langchain.utils import get_from_dict_or_env
from langchain import SerpAPIWrapper

from src.utils_langchain import _chunk_sources


class H2OSerpAPIWrapper(SerpAPIWrapper):
    def get_search_documents(self, query, chunk=True, chunk_size=512, db_type='chroma'):
        search_results_list = self.run(query, return_list=True)
        docs = [Document(page_content=x, metadata=dict(source='Web Search %s' % query, score=1.0)) for x
                in search_results_list]
        chunk_sources = functools.partial(_chunk_sources, chunk=chunk, chunk_size=chunk_size, db_type=db_type)
        docs = chunk_sources(docs)
        for xi, x in enumerate(docs):
            x.page_content = 'Web search result %d: ' % xi + x.page_content
            x.metadata['web_result_index'] = xi

        return docs

    @staticmethod
    def _process_response(res: dict, return_list=True) -> typing.Union[str, list]:
        """Process response from SerpAPI."""
        if "error" in res.keys():
            raise ValueError(f"Got error from SerpAPI: {res['error']}")
        if "answer_box_list" in res.keys():
            res["answer_box"] = res["answer_box_list"]
        if "answer_box" in res.keys():
            answer_box = res["answer_box"]
            if isinstance(answer_box, list):
                answer_box = answer_box[0]
            if "result" in answer_box.keys():
                return answer_box["result"]
            elif "answer" in answer_box.keys():
                return answer_box["answer"]
            elif "snippet" in answer_box.keys():
                return answer_box["snippet"]
            elif "snippet_highlighted_words" in answer_box.keys():
                return answer_box["snippet_highlighted_words"]
            else:
                answer = {}
                for key, value in answer_box.items():
                    if not isinstance(value, (list, dict)) and not (
                            type(value) == str and value.startswith("http")
                    ):
                        answer[key] = value
                return str(answer)
        elif "events_results" in res.keys():
            return res["events_results"][:10]
        elif "sports_results" in res.keys():
            return res["sports_results"]
        elif "top_stories" in res.keys():
            return res["top_stories"]
        elif "news_results" in res.keys():
            return res["news_results"]
        elif "jobs_results" in res.keys() and "jobs" in res["jobs_results"].keys():
            return res["jobs_results"]["jobs"]
        elif (
                "shopping_results" in res.keys()
                and "title" in res["shopping_results"][0].keys()
        ):
            return res["shopping_results"][:3]
        elif "questions_and_answers" in res.keys():
            return res["questions_and_answers"]
        elif (
                "popular_destinations" in res.keys()
                and "destinations" in res["popular_destinations"].keys()
        ):
            return res["popular_destinations"]["destinations"]
        elif "top_sights" in res.keys() and "sights" in res["top_sights"].keys():
            return res["top_sights"]["sights"]
        elif (
                "images_results" in res.keys()
                and "thumbnail" in res["images_results"][0].keys()
        ):
            return str([item["thumbnail"] for item in res["images_results"][:10]])

        snippets = []
        if "knowledge_graph" in res.keys():
            knowledge_graph = res["knowledge_graph"]
            title = knowledge_graph["title"] if "title" in knowledge_graph else ""
            if "description" in knowledge_graph.keys():
                snippets.append(knowledge_graph["description"])
            for key, value in knowledge_graph.items():
                if (
                        type(key) == str
                        and type(value) == str
                        and key not in ["title", "description"]
                        and not key.endswith("_stick")
                        and not key.endswith("_link")
                        and not value.startswith("http")
                ):
                    snippets.append(f"{title} {key}: {value}.")
        if "organic_results" in res.keys():
            for org_res in res["organic_results"]:
                if "snippet" in org_res.keys():
                    snippets.append(org_res["snippet"])
                elif "snippet_highlighted_words" in org_res.keys():
                    snippets.append(org_res["snippet_highlighted_words"])
                elif "rich_snippet" in org_res.keys():
                    snippets.append(org_res["rich_snippet"])
                elif "rich_snippet_table" in org_res.keys():
                    snippets.append(org_res["rich_snippet_table"])
                elif "link" in org_res.keys():
                    snippets.append(org_res["link"])
        if "buying_guide" in res.keys():
            snippets.append(res["buying_guide"])
        if "local_results" in res.keys() and "places" in res["local_results"].keys():
            snippets.append(res["local_results"]["places"])

        if len(snippets) > 0:
            if return_list:
                return snippets
            else:
                return '\n'.join(snippets)
        else:
            return "No good search result found"
