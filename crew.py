import os
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import JSONSearchTool

# 1. Setup Environment (Matching your Guide)
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["MODEL"] = "ollama/phi3"

@CrewBase
class AgentReviewCrew():
    """Yelp Review Prediction Crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # 2. Tool Configuration (RAG for JSON)
    # This config ensures we use local CPU for embeddings to avoid API keys
    rag_config = {
        "embedding_model": {
            "provider": "sentence-transformer",
            "config": {"model_name": "BAAI/bge-small-en-v1.5"}
        }
    }

    # Define the 3 RAG tools for the 3 datasets
    user_tool = JSONSearchTool(json_path='data/user_subset.json', config=rag_config)
    item_tool = JSONSearchTool(json_path='data/item_subset.json', config=rag_config)
    review_tool = JSONSearchTool(json_path='data/review_subset.json', config=rag_config)

    # 3. Agents Definition
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore
            tools=[self.user_tool, self.item_tool, self.review_tool],
            verbose=True,
            llm="ollama/phi3"
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore
            verbose=True,
            llm="ollama/phi3"
        )

    # 4. Tasks Definition
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AgentReview crew"""
        return Crew(
            agents=self.agents, # type: ignore
            tasks=self.tasks, # type: ignore
            process=Process.sequential,
            verbose=True,
        )