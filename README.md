# Municipal Knowledge Graph Pipeline

This project is a proof-of-concept pipeline that ingests raw transcripts from city council meetings and transforms them into a structured, interactive Knowledge Graph (KG). The goal is to create a foundation for a system that allows journalists, researchers, and organizations to easily query and analyze municipal activities.

The pipeline automates the entire process from raw text to a visualized graph, containerized with Docker for easy and reproducible execution.

---

## 🚀 How to Run the Project

The entire pipeline is containerized using Docker and can be executed with a single command.

### Prerequisites

* **Git:** To clone the repository.
* **Docker & Docker Compose:** To build and run the containerized application.

### Instructions

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/jayeshbankoti007/Municipal-Knowledge-Graph-Pipeline
    cd Municipal-Knowledge-Graph-Pipeline
    ```

2.  **Set Up Environment**
    Create a `.env` file in the root of the project and add your OpenAI API key. This is required for the entity extraction step.
    ```
    OPENAI_API_KEY="sk-..."
    ```

3.  **Place Data**
    Download the `transcripts.zip` file provided in the task description and place it inside the `data/` directory. 
    The `run.sh` script will automatically handle unzipping.

    ## 📋 What `run.sh` does

    The script will automatically:
    - ✅ Validate your setup (.env, data files, Docker)
    - ✅ Build the Docker image
    - ✅ Run the entire pipeline (extract → resolve → graph → visualize)
    - ✅ Generate outputs in `output/` directory


4.  **Execute the Pipeline**
    Run the provided shell script (e.g., `run.sh`).
    This will build the Docker image, run the pipeline, and generate the outputs in the `/output` directory.
    ```bash
    chmod +x run.sh
    ./run.sh
    ```
    This script automates the `docker-compose up --build` command, which orchestrates the entire process.

5.  **View the Outputs**
    Once the pipeline completes, the `output/` directory will contain:
    * `knowledge_graph.pkl`: A pickled NetworkX graph object.
    * `knowledge_graph.graphml`: A GraphML file compatible with graph databases like Neo4j.
    * `knowledge_graph_interactive.html`: An interactive HTML visualization of the graph. Open this file in your web browser to explore the nodes and relationships.
    * `extractions/`: A directory containing the raw JSON data extracted from each transcript.
    * `resolved_entities_dict.json`: The dictionary mapping entity aliases to their canonical forms.

---

## 🏛️ Architectural Design

The pipeline is designed to be modular, scalable, and easily expandable. It consists of three main, decoupled stages orchestrated by a main `Pipeline` class.

### 1. Preprocessing & Entity Extraction

* **Design Choice:** Instead of using traditional NLP models, this pipeline leverages a powerful Large Language Model (`gpt-4o-mini` and `gpt-4.1-nano`) via the LangChain framework for entity extraction. This approach offers high flexibility and can extract nuanced information, including relationships and predictions, which would be challenging for rule-based systems.
* **Efficiency:** To handle long transcripts and manage API costs, a **two-step preprocessing** phase is implemented in `preprocess.py`. First, a rule-based scoring system (`score_sentence`) intelligently filters out procedural chatter and retains high-value sentences. Second, a smaller, faster and cheaper LLM (`gpt-4.1-nano`) summarizes this reduced text into a dense, fact-focused summary before sending it to the more powerful extraction model.
* **Structured Output:** Finally `gpt-4o-mini` is called for final response and the output is parsed and validated into Pydantic models (`TranscriptExtraction`), ensuring a clean, structured, and reliable data format for downstream tasks.


### 2. Entity Resolution

* **Design Choice:** A pragmatic, rule-based approach was chosen for its speed, transparency, and effectiveness for this dataset. This avoids the overhead of a complex model-based solution. The `EntityResolver` class is responsible for this logic.
* **Approach:**
    * **Bill Normalization:** Bill IDs (e.g., "25-O-1271") are normalized using regular expressions to a canonical format.
    * **Fuzzy Matching:** For Organizations and Projects, where names can vary (e.g., "Dept of Finance", "Finance Department"), a fuzzy matching algorithm based on `difflib.SequenceMatcher` is used. It groups similar entities under a single canonical name based on a similarity threshold.
* **Output:** The result is a simple alias-to-canonical lookup dictionary, which is highly efficient for the final graph-building step.

### 3. Knowledge Graph Construction & Visualization

* **Design Choice:** The popular `networkx` library is used to construct the graph in memory, offering a robust and flexible foundation. The graph schema is defined implicitly through the node/edge creation logic, connecting entities like `Person`, `Bill`, `Organization`, and `Project` through relationships such as `VOTED_ON`, `MEMBER_OF`, and `AUTHORIZES`.
* **Expandability:** Adding new entities or relationships is straightforward:
    1.  Update the Pydantic models in `models.py`.
    2.  Modify the LLM prompt in `extract.py`.
    3.  Add the new node/edge logic in `graph.py`.
* **Visualization:** The graph is visualized using `Plotly`, generating a fully interactive HTML file. This allows for dynamic exploration, with hover-over tooltips providing rich details about each node and its connections, making the data highly accessible.

---

## ✨ Bonus Task: Bill Outcome Prediction

The optional task to predict bill outcomes was successfully implemented.

* **Approach:** This was integrated directly into the entity extraction step. The LLM prompt in `extract.py` explicitly instructs the model to act as a political analyst. For each bill, it must predict an outcome (`APPROVED`, `REJECTED`, or `UNCERTAIN`), provide a confidence level, and state its reasoning.
* **Contextual Factors:** The prompt guides the LLM to base its prediction on key indicators from the transcript, such as:
    * Explicit voting records.
    * The sentiment of discussions (e.g., strong support or concerns raised).
    * Procedural actions like amendments or holds.
* **Implementation:** The `Bill` Pydantic model was extended with `prediction`, `confidence`, and `reasoning` fields to capture this information structurally.
* **Visualization:** This prediction is a key feature of the interactive graph. When a user hovers over a Bill node (highlighted in gold), the tooltip prominently displays the predicted outcome and the LLM's reasoning, providing immediate insight.
