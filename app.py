import os
import time
from collections import deque

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from google import genai


# =========================================================
# 1. PAGE CONFIGURATION AND SETTINGS
# =========================================================
st.set_page_config(
    page_title="SupplyLink AI",
    page_icon="📦",
    layout="wide"
)

load_dotenv()

st.title("📦 SupplyLink AI")
st.caption("Multi-Warehouse Supply Chain Planner | AI learning and decision-support prototype")


# =========================================================
# 2. SAMPLE INVENTORY DATA
# =========================================================
def load_inventory():
    data = {
        "Product": [
            "Laptop", "Mouse", "Keyboard", "Monitor", "Printer",
            "Laptop", "Mouse", "Keyboard", "Monitor", "Printer",
            "Laptop", "Mouse", "Keyboard", "Monitor", "Printer"
        ],
        "Warehouse": [
            "Bangalore", "Bangalore", "Bangalore", "Bangalore", "Bangalore",
            "Mumbai", "Mumbai", "Mumbai", "Mumbai", "Mumbai",
            "Delhi", "Delhi", "Delhi", "Delhi", "Delhi"
        ],
        "Stock": [
            10, 100, 20, 15, 5,
            50, 20, 80, 10, 30,
            25, 70, 15, 60, 10
        ],
        "Minimum Stock": [
            20, 30, 25, 20, 10,
            20, 30, 25, 20, 10,
            20, 30, 25, 20, 10
        ]
    }

    frame = pd.DataFrame(data)
    frame["Shortage"] = (frame["Minimum Stock"] - frame["Stock"]).clip(lower=0)
    frame["Surplus"] = (frame["Stock"] - frame["Minimum Stock"]).clip(lower=0)
    frame["Status"] = frame.apply(
        lambda row: "Low Stock" if row["Stock"] < row["Minimum Stock"] else "Available",
        axis=1
    )
    return frame


if "inventory" not in st.session_state:
    st.session_state.inventory = load_inventory()

df = st.session_state.inventory.copy()


# =========================================================
# 3. GEMINI API
# =========================================================
def get_api_key():
    """Read the key locally from .env or remotely from Streamlit Secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def get_gemini_response(prompt):
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found. Add it to your local .env file "
            "or Streamlit Cloud Settings > Secrets."
        )

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 60000}
    )

    start_time = time.perf_counter()
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "max_output_tokens": 400,
            "temperature": 0.3
        }
    )
    elapsed = time.perf_counter() - start_time

    if not response.text:
        raise ValueError("Gemini returned an empty response.")
    return response.text, elapsed


# =========================================================
# 4. REUSABLE INVENTORY / AI FUNCTIONS
# =========================================================
def create_transfer_recommendations(inventory):
    recommendations = []

    for product in inventory["Product"].unique():
        product_data = inventory[inventory["Product"] == product]
        donors = product_data[product_data["Surplus"] > 0].copy()
        receivers = product_data[product_data["Shortage"] > 0].copy()

        for receiver_index, receiver in receivers.iterrows():
            remaining_shortage = int(receiver["Shortage"])

            for donor_index, donor in donors.iterrows():
                if remaining_shortage <= 0:
                    break
                if donor["Warehouse"] == receiver["Warehouse"]:
                    continue

                available_surplus = int(donors.loc[donor_index, "Surplus"])
                quantity = min(remaining_shortage, available_surplus)

                if quantity > 0:
                    recommendations.append({
                        "Product": product,
                        "From Warehouse": donor["Warehouse"],
                        "To Warehouse": receiver["Warehouse"],
                        "Transfer Quantity": quantity
                    })
                    remaining_shortage -= quantity
                    donors.loc[donor_index, "Surplus"] -= quantity

    return pd.DataFrame(recommendations)


def graph_search(graph, start, goal, method="BFS"):
    """Demonstration search over a small, illustrative warehouse connection graph."""
    frontier = deque([[start]])
    visited = set()

    while frontier:
        path = frontier.popleft() if method == "BFS" else frontier.pop()
        node = path[-1]

        if node == goal:
            return path

        if node not in visited:
            visited.add(node)
            neighbors = graph.get(node, [])
            for neighbor in neighbors:
                if neighbor not in visited:
                    frontier.append(path + [neighbor])

    return None


def run_inventory_agent(inventory, product):
    """A transparent rule-based agent: observe -> decide -> recommend."""
    product_rows = inventory[inventory["Product"] == product]
    observations = []
    actions = []

    if product_rows.empty:
        return ["No inventory records found for this product."], []

    for _, row in product_rows.iterrows():
        observations.append(
            f'{row["Warehouse"]}: stock={int(row["Stock"])}, '
            f'minimum={int(row["Minimum Stock"])}, '
            f'shortage={int(row["Shortage"])}, surplus={int(row["Surplus"])}'
        )
        if row["Shortage"] > 0:
            actions.append(
                f'Flag {row["Warehouse"]} for replenishment '
                f'({int(row["Shortage"])} units below minimum).'
            )
        elif row["Surplus"] > 0:
            actions.append(
                f'{row["Warehouse"]} may be a donor for up to '
                f'{int(row["Surplus"])} units above its minimum.'
            )
        else:
            actions.append(f'{row["Warehouse"]} is at its minimum stock level.')

    return observations, actions


# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    [
        "Dashboard",
        "Inventory",
        "Transfer Recommendations",
        "AI Assistant",
        "AI Learning Modules",
        "Reports"
    ]
)


# =========================================================
# 6. DASHBOARD
# =========================================================
if page == "Dashboard":
    st.header("Inventory Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Product-Warehouse Records", len(df))
    col2.metric("Total Stock Units", int(df["Stock"].sum()))
    col3.metric("Low-Stock Records", int((df["Status"] == "Low Stock").sum()))
    col4.metric("Warehouses", df["Warehouse"].nunique())

    left, right = st.columns(2)
    with left:
        st.subheader("Stock by Warehouse")
        st.bar_chart(df.groupby("Warehouse")["Stock"].sum())
    with right:
        st.subheader("Inventory Status")
        st.bar_chart(df["Status"].value_counts())

    st.subheader("Inventory Overview")
    st.dataframe(df, use_container_width=True)


# =========================================================
# 7. INVENTORY PAGE
# =========================================================
elif page == "Inventory":
    st.header("Inventory Management")

    search = st.text_input("Search Product")
    warehouse_filter = st.selectbox(
        "Select Warehouse",
        ["All"] + sorted(df["Warehouse"].unique().tolist())
    )

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["Product"].str.contains(search, case=False, na=False)]
    if warehouse_filter != "All":
        filtered = filtered[filtered["Warehouse"] == warehouse_filter]

    st.dataframe(filtered, use_container_width=True)

    st.subheader("Low-Stock Products")
    low_stock = df[df["Shortage"] > 0]
    if low_stock.empty:
        st.success("No low-stock products.")
    else:
        st.dataframe(low_stock, use_container_width=True)


# =========================================================
# 8. TRANSFER RECOMMENDATIONS
# =========================================================
elif page == "Transfer Recommendations":
    st.header("Warehouse Transfer Recommendations")
    st.write(
        "This baseline decision-support rule matches surplus at one warehouse "
        "with shortages at another. It does not calculate actual distance, "
        "transportation cost, or live delivery availability."
    )

    transfer_df = create_transfer_recommendations(df)
    if not transfer_df.empty:
        st.success("Potential transfers generated from the sample inventory.")
        st.dataframe(transfer_df, use_container_width=True)
        csv = transfer_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Transfer Recommendations",
            data=csv,
            file_name="transfer_recommendations.csv",
            mime="text/csv"
        )
    else:
        st.info("No suitable warehouse transfers found.")


# =========================================================
# 9. GEMINI AI ASSISTANT
# =========================================================
elif page == "AI Assistant":
    st.header("SupplyLink AI Assistant")
    st.write("Ask Gemini to analyze the inventory data and explain its recommendations.")

    with st.form("gemini_form"):
        prompt = st.text_input(
            "Enter your question",
            placeholder="Which products have low stock?"
        )
        submit = st.form_submit_button("Analyze Inventory")

    if submit:
        if not prompt.strip():
            st.warning("Please enter a question.")
        else:
            inventory_context = df[
                ["Product", "Warehouse", "Stock", "Minimum Stock", "Shortage", "Surplus", "Status"]
            ].to_string(index=False)

            full_prompt = f"""
You are SupplyLink AI, a supply-chain decision-support assistant.
Use only the inventory data below for numerical claims.
Do not claim that a recommendation was executed; recommendations are suggestions.

INVENTORY DATA:
{inventory_context}

USER QUESTION:
{prompt}

Give a clear answer with short bullet points where useful.
Mention limitations if the data is insufficient.
"""
            try:
                with st.spinner("Gemini is analyzing your inventory..."):
                    answer, elapsed = get_gemini_response(full_prompt)
                st.success(f"Analysis completed in {elapsed:.2f} seconds.")
                st.subheader("AI Analysis")
                st.markdown(answer)
            except Exception as error:
                st.error("Gemini request failed.")
                st.code(f"{type(error).__name__}: {error}")


# =========================================================
# 10. FIVE AI LEARNING MODULES
# =========================================================
elif page == "AI Learning Modules":
    st.header("AI Learning Modules")
    st.caption(
        "Five small, transparent demonstrations aligned with the learning roadmap. "
        "These are educational prototypes, not production logistics optimizers."
    )

    module = st.selectbox(
        "Choose a module",
        [
            "Phase 1 - Introduction and Searching Decisions",
            "Phase 2 - Constraint Satisfaction Problems",
            "Phase 3 - Introduction to AI Agents",
            "Phase 4 - Frameworks for Building AI Agents",
            "Phase 5 - Ethics, Authentic Intelligence and AI Trends",
            "Module Coverage"
        ]
    )

    if module.startswith("Phase 1"):
        st.subheader("Phase 1: Introduction and Searching Decisions")
        st.write(
            "Search algorithms explore connected states to find a path. "
            "This example uses an illustrative connection graph between warehouses."
        )
        graph = {
            "Bangalore": ["Mumbai", "Delhi"],
            "Mumbai": ["Bangalore", "Delhi"],
            "Delhi": ["Bangalore", "Mumbai"]
        }
        c1, c2, c3 = st.columns(3)
        start = c1.selectbox("Start warehouse", list(graph.keys()), key="search_start")
        goal = c2.selectbox("Goal warehouse", list(graph.keys()), index=1, key="search_goal")
        method = c3.selectbox("Search method", ["BFS", "DFS"])

        if st.button("Run Search", key="run_search"):
            if start == goal:
                st.info("Start and goal are the same warehouse. Path contains one node.")
                st.success(start)
            else:
                path = graph_search(graph, start, goal, method)
                if path:
                    st.success("Search path: " + " → ".join(path))
                    st.write(f"Number of edges in this path: {len(path) - 1}")
                else:
                    st.warning("No path found.")

        st.info(
            "Coverage: state-space representation, graph traversal, and a comparison "
            "of Breadth-First Search (BFS) and Depth-First Search (DFS). "
            "The graph is illustrative and does not represent measured road distances."
        )

    elif module.startswith("Phase 2"):
        st.subheader("Phase 2: Constraint Satisfaction Problems (CSP)")
        st.write(
            "A CSP has variables, domains, and constraints. Here, transfer quantity "
            "is limited by donor surplus, receiver shortage, and receiver capacity."
        )

        transfer_df = create_transfer_recommendations(df)
        if transfer_df.empty:
            st.info("No candidate transfers are available in the current sample.")
        else:
            selected_index = st.selectbox(
                "Choose a candidate transfer",
                transfer_df.index.tolist(),
                format_func=lambda index: (
                    f'{transfer_df.loc[index, "Product"]}: '
                    f'{transfer_df.loc[index, "From Warehouse"]} → '
                    f'{transfer_df.loc[index, "To Warehouse"]}'
                )
            )
            candidate = transfer_df.loc[selected_index]
            proposed = int(candidate["Transfer Quantity"])
            receiver_stock = int(df[
                (df["Product"] == candidate["Product"]) &
                (df["Warehouse"] == candidate["To Warehouse"])
            ]["Stock"].iloc[0])
            receiver_min = int(df[
                (df["Product"] == candidate["Product"]) &
                (df["Warehouse"] == candidate["To Warehouse"])
            ]["Minimum Stock"].iloc[0])

            capacity = st.number_input(
                "Illustrative maximum capacity for this receiver warehouse/product",
                min_value=0,
                max_value=1000,
                value=max(receiver_stock + proposed, receiver_min),
                step=1
            )
            capacity_room = max(0, int(capacity) - receiver_stock)
            feasible_qty = min(proposed, capacity_room)

            st.write(f"Candidate transfer quantity: **{proposed}**")
            st.write(f"Receiver current stock: **{receiver_stock}**")
            st.write(f"Receiver minimum stock: **{receiver_min}**")
            st.write(f"Capacity available: **{capacity_room}**")

            if feasible_qty > 0:
                st.success(f"Constraint check passed for up to {feasible_qty} units.")
            else:
                st.error("Constraint check failed: no capacity is available.")

            st.caption(
                "Simplified CSP demonstration: variable = transfer quantity; "
                "domain = 0 to proposed quantity; constraints = donor surplus, "
                "receiver shortage, and receiver capacity."
            )

    elif module.startswith("Phase 3"):
        st.subheader("Phase 3: Introduction to AI Agents")
        st.write(
            "An agent observes its environment, selects an action based on its policy, "
            "and presents an output. This demo uses transparent inventory rules."
        )
        product = st.selectbox("Choose a product", sorted(df["Product"].unique()), key="agent_product")
        observations, actions = run_inventory_agent(df, product)

        st.markdown("**1. Observe — inventory facts**")
        for item in observations:
            st.write("• " + item)

        st.markdown("**2. Decide — rule-based policy output**")
        for item in actions:
            st.write("• " + item)

        st.markdown("**3. Act — proposed next step**")
        st.info(
            "The agent presents recommendations for a human to review. "
            "It does not modify stock or place orders automatically."
        )

    elif module.startswith("Phase 4"):
        st.subheader("Phase 4: Frameworks for Building AI Agents")
        st.write(
            "This module explains the building blocks commonly used by agent frameworks "
            "and demonstrates a small local tool-routing pattern."
        )

        st.markdown("**Agent architecture building blocks**")
        architecture = pd.DataFrame({
            "Component": ["Input", "Planner / policy", "Tools", "Memory / state", "Output / guardrail"],
            "SupplyLink AI example": [
                "User asks about a product",
                "Choose an inventory analysis task",
                "Read inventory, calculate shortage, find transfers",
                "Streamlit session state holds sample inventory",
                "Show recommendation and require human review"
            ]
        })
        st.dataframe(architecture, use_container_width=True)

        question_type = st.selectbox(
            "Select a task to route",
            ["Find low-stock records", "Suggest transfers", "Summarize warehouse stock"]
        )
        if st.button("Run local tool router", key="tool_router"):
            if question_type == "Find low-stock records":
                result = df[df["Shortage"] > 0][
                    ["Product", "Warehouse", "Stock", "Minimum Stock", "Shortage"]
                ]
                st.write("Tool selected: `low_stock_tool`")
                st.dataframe(result, use_container_width=True)
            elif question_type == "Suggest transfers":
                st.write("Tool selected: `transfer_recommendation_tool`")
                st.dataframe(create_transfer_recommendations(df), use_container_width=True)
            else:
                st.write("Tool selected: `warehouse_summary_tool`")
                st.dataframe(
                    df.groupby("Warehouse", as_index=False)["Stock"].sum(),
                    use_container_width=True
                )

        st.warning(
            "This is a lightweight educational tool-router built in Python, "
            "not an implementation using LangChain, LangGraph, CrewAI, or another "
            "third-party agent framework. Add one of those frameworks only if your "
            "assignment explicitly requires a named framework."
        )

    elif module.startswith("Phase 5"):
        st.subheader("Phase 5: Ethics, Authentic Intelligence and Trends in AI")
        st.write("Review these responsible-AI principles before using recommendations.")
        principles = [
            ("Human oversight", "A person reviews transfers before inventory is changed."),
            ("Transparency", "Show the stock values and rules behind each recommendation."),
            ("Privacy and security", "Never publish API keys or private business inventory data."),
            ("Reliability", "Validate input data and clearly show when information is missing."),
            ("Fairness", "Avoid prioritizing a warehouse or customer without a justified business rule."),
            ("Accountability", "Record who approved an action and why when adding a production workflow.")
        ]
        for heading, detail in principles:
            st.markdown(f"**{heading}:** {detail}")

        st.markdown("**Limitations of this prototype**")
        st.write(
            "- Uses a small, hard-coded sample dataset.\n"
            "- Does not connect to live warehouse systems.\n"
            "- Does not calculate actual transport routes or costs.\n"
            "- Gemini output can be incorrect and must be checked.\n"
            "- Recommendations are not automatically executed."
        )

        ethics_ack = st.checkbox(
            "I understand that AI outputs are suggestions and require human verification."
        )
        if ethics_ack:
            st.success("Responsible-use acknowledgement recorded for this session.")

    else:
        st.subheader("Module Coverage Matrix")
        coverage = pd.DataFrame([
            {
                "Roadmap module": "Phase 1 - Searching Decisions",
                "Demonstration in app": "BFS and DFS on an illustrative warehouse graph",
                "Evidence": "Run Search result and displayed path",
                "Status": "Demo implemented; document theory and complexity"
            },
            {
                "Roadmap module": "Phase 2 - CSP and AI Reasoning",
                "Demonstration in app": "Transfer quantity constrained by surplus, shortage, capacity",
                "Evidence": "Constraint check result",
                "Status": "Simplified CSP demo implemented"
            },
            {
                "Roadmap module": "Phase 3 - AI Agents",
                "Demonstration in app": "Observe, decide, recommend inventory agent",
                "Evidence": "Observations and proposed actions",
                "Status": "Rule-based agent demo implemented"
            },
            {
                "Roadmap module": "Phase 4 - Agent Frameworks",
                "Demonstration in app": "Local tool router and agent architecture table",
                "Evidence": "Selected tool and tool output",
                "Status": "Framework concepts demo; no third-party framework installed"
            },
            {
                "Roadmap module": "Phase 5 - AI Ethics and Trends",
                "Demonstration in app": "Human oversight, transparency, privacy, reliability",
                "Evidence": "Ethics checklist and limitations",
                "Status": "Ethics section implemented; research references may be added"
            }
        ])
        st.dataframe(coverage, use_container_width=True)
        st.caption(
            "Use this matrix as an implementation tracker. Complete your written "
            "explanations, test evidence, and any instructor-specific requirements separately."
        )


# =========================================================
# 11. REPORTS
# =========================================================
elif page == "Reports":
    st.header("Inventory Reports")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Stock by Product")
        st.bar_chart(df.groupby("Product")["Stock"].sum())
    with col2:
        st.subheader("Stock by Warehouse")
        st.bar_chart(df.groupby("Warehouse")["Stock"].sum())

    st.subheader("Low Stock Report")
    st.dataframe(df[df["Shortage"] > 0], use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Inventory CSV",
        data=csv,
        file_name="supplylink_inventory.csv",
        mime="text/csv"
    )
    st.success("Reports are ready.")


# =========================================================
# 12. FOOTER
# =========================================================
st.sidebar.divider()
st.sidebar.caption("SupplyLink AI | Educational multi-warehouse decision-support prototype")
