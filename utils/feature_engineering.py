import pandas as pd
from datetime import datetime


def prepare_features(
    shipment_type,
    scheduled_days,
    benefit,
    sales,
    category,
    customer_segment,
    department,
    market,
    discount,
    discount_rate,
    product_price,
    quantity,
    profit,
    order_region,
    shipping_mode
):

    # -------------------------
    # Basic Calculations
    # -------------------------

    sales_per_customer = sales

    order_item_total = sales - discount

    if sales != 0:
        profit_ratio = profit / sales
    else:
        profit_ratio = 0

    current_date = datetime.today()

    order_month = current_date.month
    order_year = current_date.year
    order_quarter = ((order_month - 1) // 3) + 1
    order_day_of_week = current_date.weekday()

    weekend_order = 1 if order_day_of_week >= 5 else 0

    peak_season = 1 if order_month in [11, 12] else 0

    high_profit_order = 1 if profit >= 20 else 0

    discounted_order = 1 if discount > 0 else 0

    high_value_order = 1 if sales >= 500 else 0

    priority_customer = (
        1 if customer_segment in
        ["Corporate", "Home Office"]
        else 0
    )

    # -------------------------
    # Categories
    # -------------------------

    if sales < 100:
        order_value_category = "Low Value"
    elif sales < 500:
        order_value_category = "Medium Value"
    else:
        order_value_category = "High Value"

    if quantity == 1:
        quantity_category = "Single Item"
    elif quantity <= 5:
        quantity_category = "Medium Quantity"
    else:
        quantity_category = "Bulk Order"

    if profit < 0:
        profit_category = "Loss"
    elif profit < 10:
        profit_category = "Low Margin"
    elif profit < 20:
        profit_category = "Medium Margin"
    else:
        profit_category = "High Margin"

    # -------------------------
    # Final DataFrame
    # -------------------------

    data = pd.DataFrame({
        "Type": [shipment_type],
        "Days for shipment (scheduled)": [scheduled_days],
        "Benefit per order": [benefit],
        "Sales per customer": [sales_per_customer],
        "Category Name": [category],
        "Customer Segment": [customer_segment],
        "Department Name": [department],
        "Market": [market],
        "Order Item Discount": [discount],
        "Order Item Discount Rate": [discount_rate],
        "Order Item Product Price": [product_price],
        "Order Item Profit Ratio": [profit_ratio],
        "Order Item Quantity": [quantity],
        "Sales": [sales],
        "Order Item Total": [order_item_total],
        "Order Profit Per Order": [profit],
        "Order Region": [order_region],
        "Product Price": [product_price],
        "Shipping Mode": [shipping_mode],
        "Order_Month": [order_month],
        "Weekend_Order": [weekend_order],
        "Order_Year": [order_year],
        "High_Profit_Order": [high_profit_order],
        "Discounted_Order": [discounted_order],
        "Order_Value_Category": [order_value_category],
        "High_Value_Order": [high_value_order],
        "Quantity_Category": [quantity_category],
        "Profit_Category": [profit_category],
        "Priority_Customer": [priority_customer],
        "Order_Quarter": [order_quarter],
        "Order_Day_of_Week": [order_day_of_week],
        "Peak_Season": [peak_season]
    })

    return data

def prepare_bulk_features(df):

    data = df.copy()

    # ------------------------------------------------
    # Weekend Order
    # ------------------------------------------------

    data["Weekend_Order"] = (
        data["Order Weekday"]
        .isin(["Saturday", "Sunday"])
        .astype(int)
    )

    # ------------------------------------------------
    # High Profit Order
    # ------------------------------------------------

    data["High_Profit_Order"] = (
        data["Order Profit Per Order"] > 0
    ).astype(int)

    # ------------------------------------------------
    # Discounted Order
    # ------------------------------------------------

    data["Discounted_Order"] = (
        data["Order Item Discount"] > 0
    ).astype(int)

    # ------------------------------------------------
    # Order Value Category
    # ------------------------------------------------

    data["Order_Value_Category"] = pd.cut(

        data["Sales"],

        bins=[0, 200, 500, float("inf")],

        labels=[
            "Low Value",
            "Medium Value",
            "High Value"
        ]

    )

    # ------------------------------------------------
    # High Value Order
    # ------------------------------------------------

    data["High_Value_Order"] = (
        data["Sales"] >= 500
    ).astype(int)

    # ------------------------------------------------
    # Quantity Category
    # ------------------------------------------------

    data["Quantity_Category"] = np.where(

        data["Order Item Quantity"] == 1,

        "Single Item",

        np.where(

            data["Order Item Quantity"] <= 5,

            "Medium Quantity",

            "Bulk Order"

        )

    )

    # ------------------------------------------------
    # Profit Category
    # ------------------------------------------------

    data["Profit_Category"] = np.select(

        [

            data["Order Item Profit Ratio"] < 0,

            data["Order Item Profit Ratio"] < 0.15,

            data["Order Item Profit Ratio"] < 0.35,

            data["Order Item Profit Ratio"] >= 0.35

        ],

        [

            "Loss",

            "Low Margin",

            "Medium Margin",

            "High Margin"

        ]

    )

    # ------------------------------------------------
    # Priority Customer
    # ------------------------------------------------

    data["Priority_Customer"] = (

        data["Customer Segment"]

        .isin(

            [

                "Corporate",

                "Home Office"

            ]

        )

    ).astype(int)

    # ------------------------------------------------
    # Peak Season
    # ------------------------------------------------

    data["Peak_Season"] = (

        data["Order Month"]

        .isin(

            [11, 12]

        )

    ).astype(int)

    # ------------------------------------------------
    # Rename columns
    # ------------------------------------------------

    data.rename(

        columns={

            "Order Month": "Order_Month",

            "Order Year": "Order_Year",

            "Order Quarter": "Order_Quarter",

            "Order Weekday": "Order_Day_of_Week"

        },

        inplace=True

    )

    return data