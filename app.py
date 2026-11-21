
import os
import time
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from google import genai


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SupplyLink AI",
    page_icon="📦",
    layout="wide"
)

load_dotenv()

st.title("📦 SupplyLink AI")
st.caption("Multi-Warehouse Supply Chain Planner")


# ==========================================
# 2. SAMPLE INVENTORY DATA
# ==========================================

def load_inventory():

    data = {
        "Product": [
            "Laptop", "Mouse", "Keyboard", "Monitor", "Printer",
            "Laptop", "Mouse", "Keyboard", "Monitor", "Printer",
            "Laptop", "Mouse", "Keyboard", "Monitor", "Printer"
        ],

        "Warehouse": [
            "Bangalore", "Bangalore", "Bangalore",
            "Bangalore", "Bangalore",
            "Mumbai", "Mumbai", "Mumbai",
            "Mumbai", "Mumbai",
            "Delhi", "Delhi", "Delhi",
            "Delhi", "Delhi"
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

    df = pd.DataFrame(data)

    df["Shortage"] = (
        df["Minimum Stock"] - df["Stock"]
    ).clip(lower=0)

    df["Surplus"] = (
        df["Stock"] - df["Minimum Stock"]
    ).clip(lower=0)

    df["Status"] = df["Stock"].apply(
        lambda x: "Low Stock" if x < 20 else "Available"
    )

    return df


# Keep inventory data in session state
if "inventory" not in st.session_state:
    st.session_state.inventory = load_inventory()

df = st.session_state.inventory


# ==========================================
# 3. GEMINI API FUNCTION
# ==========================================

def get_gemini_response(prompt):

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. "
            "Check your .env file."
        )

    client = genai.Client(
        api_key=api_key,
        http_options={"timeout": 20000}
    )

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "max_output_tokens": 300,
            "temperature": 0.3
        }
    )

    elapsed = time.perf_counter() - start_time

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response.text, elapsed


# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Dashboard",
        "Inventory",
        "Transfer Recommendations",
        "AI Assistant",
        "Reports"
    ]
)


# ==========================================
# 5. DASHBOARD
# ==========================================

if page == "Dashboard":

    st.header("Inventory Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Products",
        len(df)
    )

    col2.metric(
        "Total Stock",
        int(df["Stock"].sum())
    )

    col3.metric(
        "Low Stock Items",
        int((df["Status"] == "Low Stock").sum())
    )

    col4.metric(
        "Warehouses",
        df["Warehouse"].nunique()
    )

    st.subheader("Stock by Warehouse")

    warehouse_stock = df.groupby(
        "Warehouse"
    )["Stock"].sum()

    st.bar_chart(warehouse_stock)

    st.subheader("Inventory Status")

    status_count = df["Status"].value_counts()

    st.bar_chart(status_count)

    st.subheader("Inventory Overview")

    st.dataframe(
        df,
        use_container_width=True
    )


# ==========================================
# 6. INVENTORY PAGE
# ==========================================

elif page == "Inventory":

    st.header("Inventory Management")

    search = st.text_input(
        "Search Product"
    )

    filtered_df = df.copy()

    if search:
        filtered_df = filtered_df[
            filtered_df["Product"].str.contains(
                search,
                case=False,
                na=False
            )
        ]

    warehouse_filter = st.selectbox(
        "Select Warehouse",
        ["All"] + sorted(df["Warehouse"].unique())
    )

    if warehouse_filter != "All":
        filtered_df = filtered_df[
            filtered_df["Warehouse"] == warehouse_filter
        ]

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

    st.subheader("Low Stock Products")

    low_stock = df[df["Shortage"] > 0]

    if low_stock.empty:
        st.success("No low-stock products.")

    else:
        st.dataframe(
            low_stock,
            use_container_width=True
        )


# ==========================================
# 7. TRANSFER RECOMMENDATIONS
# ==========================================

elif page == "Transfer Recommendations":

    st.header("Warehouse Transfer Recommendations")

    st.write(
        "Find warehouses with surplus stock "
        "and warehouses with shortages."
    )

    recommendations = []

    for product in df["Product"].unique():

        product_data = df[
            df["Product"] == product
        ]

        donors = product_data[
            product_data["Surplus"] > 0
        ].copy()

        receivers = product_data[
            product_data["Shortage"] > 0
        ].copy()

        for receiver_index, receiver in receivers.iterrows():

            remaining_shortage = int(receiver["Shortage"])

            for donor_index, donor in donors.iterrows():

                if remaining_shortage <= 0:
                    break

                if donor["Warehouse"] == receiver["Warehouse"]:
                    continue

                available_surplus = int(donors.loc[
                    donor_index, "Surplus"
                ])

                transfer_qty = min(
                    remaining_shortage,
                    available_surplus
                )

                if transfer_qty > 0:

                    recommendations.append({
                        "Product": product,
                        "From Warehouse": donor["Warehouse"],
                        "To Warehouse": receiver["Warehouse"],
                        "Transfer Quantity": transfer_qty
                    })

                    remaining_shortage -= transfer_qty

                    donors.loc[
                        donor_index, "Surplus"
                    ] -= transfer_qty

    if recommendations:

        transfer_df = pd.DataFrame(recommendations)

        st.success("Transfer recommendations generated!")

        st.dataframe(
            transfer_df,
            use_container_width=True
        )

        csv = transfer_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Transfer Recommendations",
            csv,
            "transfer_recommendations.csv",
            "text/csv"
        )

    else:

        st.info(
            "No suitable warehouse transfers found."
        )


# ==========================================
# 8. GEMINI AI ASSISTANT
# ==========================================

elif page == "AI Assistant":

    st.header("SupplyLink AI Assistant")

    st.write(
        "Ask Gemini to analyze inventory and "
        "provide warehouse management suggestions."
    )

    with st.form("gemini_form"):

        prompt = st.text_input(
            "Enter your question",
            placeholder="Analyze low-stock products..."
        )

        submit = st.form_submit_button(
            "Analyze Inventory"
        )

    if submit:

        if not prompt.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            inventory_context = df.to_string(index=False)

            full_prompt = f"""
            You are SupplyLink AI, an inventory and
            supply chain assistant.

            Analyze the following inventory data:

            {inventory_context}

            User question:
            {prompt}

            Give a clear, practical answer.
            Use short bullet points where appropriate.
            Do not invent inventory values.
            """

            try:

                with st.spinner(
                    "Gemini is analyzing your inventory..."
                ):

                    answer, elapsed = get_gemini_response(
                        full_prompt
                    )

                st.success(
                    f"Analysis completed in {elapsed:.2f} seconds!"
                )

                st.subheader("AI Analysis")

                st.markdown(answer)

            except Exception as e:

                st.error(
                    "Gemini API request failed or timed out."
                )

                st.exception(e)


# ==========================================
# 9. REPORTS PAGE
# ==========================================

elif page == "Reports":

    st.header("Inventory Reports")

    st.subheader("Stock by Product")

    product_stock = df.groupby(
        "Product"
    )["Stock"].sum()

    st.bar_chart(product_stock)

    st.subheader("Stock by Warehouse")

    warehouse_stock = df.groupby(
        "Warehouse"
    )["Stock"].sum()

    st.bar_chart(warehouse_stock)

    st.subheader("Download Inventory Report")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Inventory CSV",
        data=csv,
        file_name="supplylink_inventory.csv",
        mime="text/csv"
    )

    st.success("Reports are ready!")


# ==========================================
# 10. FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "SupplyLink AI | Multi-Warehouse Supply Chain Planner"
)
