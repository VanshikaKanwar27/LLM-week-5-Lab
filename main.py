import os
from crewai import Agent, Task, Crew
from crewai_tools import JSONSearchTool

# 1. Environment Configuration - CHANGED TO LLAMA 3.2:1B
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["MODEL"] = "ollama/llama3.2:1b"

# 2. Configure the RAG Tools
rag_config = {
    "embedding_model": {
        "provider": "sentence-transformer",
        "config": {"model_name": "BAAI/bge-small-en-v1.5"}
    }
}

# Ensure these paths match your folder structure exactly!
user_tool = JSONSearchTool(json_path='data/user_subset.json', config=rag_config, name="user_search")
item_tool = JSONSearchTool(json_path='data/item_subset.json', config=rag_config, name="item_search")
review_tool = JSONSearchTool(json_path='data/review_subset.json', config=rag_config, name="review_search")

# 3. Define the Agents - UPDATED LLM TO LLAMA 3.2:1B
researcher = Agent(
    role='Yelp Data Researcher',
    goal='Retrieve profile for user {user_id} and details for item {item_id}',
    backstory='Expert at finding specific data points in large Yelp datasets.',
    tools=[user_tool, item_tool, review_tool],
    verbose=True,
    llm="ollama/llama3.2:1b"
)

analyst = Agent(
    role='Rating Prediction Analyst',
    goal='Predict the star rating and write a review for the user-item pair.',
    backstory='Specializes in predicting user satisfaction based on historical data.',
    verbose=True,
    llm="ollama/llama3.2:1b"
)

# 4. Define the Tasks
research_task = Task(
    description='Search for user {user_id} and item {item_id}. Summarize their history and features.',
    expected_output='A summary of user preferences and item attributes.',
    agent=researcher
)

prediction_task = Task(
    description='Based on the research, predict the stars (1-5) and write a realistic review.',
    expected_output='A JSON object with "stars" and "review" keys.',
    agent=analyst,
    context=[research_task]
)

# 5. Execute the Crew
if __name__ == "__main__":
    yelp_crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, prediction_task],
        verbose=True,
        memory=False # Keep this False to save data/bandwidth
    )

    inputs = {
        "user_id": "LQUk3WFBgEfwIYkNDh5l1Q",
        "item_id": "KueYmi7Vrr0Hyt0_iIux4Q"
    }

    print(f"### Starting Prediction for User: {inputs['user_id']} ###")
    result = yelp_crew.kickoff(inputs=inputs)
    
    print("\n--- FINAL LAB RESULT ---")
    print(result)