import streamlit as st
import pandas as pd
import sqlite3
import datetime
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

st.set_page_config(page_title="🍽 Restaurant ", layout="wide")


st.markdown("""
<style>
.block-container {padding: 2rem;}
.stButton>button {border-radius:10px;height:45px;font-weight:bold;}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#1e1e1e;
    box-shadow:0 4px 10px rgba(0,0,0,0.3);
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)


def get_db():
    return sqlite3.connect("app.db", check_same_thread=False)

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS menu(
        item TEXT UNIQUE,
        category TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS reviews(
        food TEXT,
        review TEXT,
        sentiment INTEGER,
        rating INTEGER,
        timestamp TEXT
    )
    """)

    # default items
    c.execute("SELECT COUNT(*) FROM menu")
    if c.fetchone()[0] == 0:
        items = [
            ("Idly","Meals"),("Dosa","Meals"),
            ("Biryani","Meals"),("Noodles","Meals"),
            ("Ice Cream","Desserts"),("Juice","Drinks")
        ]
        c.executemany("INSERT INTO menu VALUES (?,?)", items)

    conn.commit()
    conn.close()

init_db()


def get_menu():
    conn = get_db()
    df = pd.read_sql("SELECT * FROM menu", conn)
    conn.close()
    return df


@st.cache_resource
def train_model():
    data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
    corpus = []

    for i in range(1000):
        r = re.sub('[^a-zA-Z]', ' ', data['Review'][i]).lower().split()
        ps = PorterStemmer()
        sw = stopwords.words('english')
        if 'not' in sw:
            sw.remove('not')
        r = [ps.stem(w) for w in r if w not in sw]
        corpus.append(' '.join(r))

    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = data.iloc[:, -1].values

    model = GaussianNB()
    model.fit(X, y)
    return model, cv

model, cv = train_model()

def predict(text):
    r = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    ps = PorterStemmer()
    sw = stopwords.words('english')
    if 'not' in sw:
        sw.remove('not')

    r = [ps.stem(w) for w in r if w not in sw]
    r = ' '.join(r)

    X = cv.transform([r]).toarray()
    res = model.predict(X)[0]

    if "not" in r:
        res = abs(res - 1)

    return res


if "owner" not in st.session_state:
    st.session_state.owner = False


st.title("🍽 Restaurant AI System")

menu_choice = st.sidebar.radio("Navigation", ["Customer", "Owner"])



if menu_choice == "Customer":

    st.markdown("### 🧾 Give Your Review")

    df_menu = get_menu()

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("📦 Category", df_menu["category"].unique())

    with col2:
        items = df_menu[df_menu["category"] == category]["item"]
        food = st.selectbox("🍽 Select Item", items)

    review = st.text_area("✍️ Write your review")
    rating = st.slider("⭐ Rating", 1, 5)

    if st.button("Submit Review"):
        sentiment = predict(review)

        conn = get_db()
        c = conn.cursor()
        c.execute("INSERT INTO reviews VALUES (?,?,?,?,?)",
                  (food, review, sentiment, rating,
                   str(datetime.datetime.now())))
        conn.commit()
        conn.close()

        if sentiment == 1:
            st.success("✅ Positive Review Saved")
        else:
            st.error("❌ Negative Review Saved")



elif menu_choice == "Owner":

    if not st.session_state.owner:

        st.markdown("### 🔐 Owner Access")
        code = st.text_input("Enter Passcode", type="password")

        if st.button("Login"):
            if code == "Roushan":
                st.session_state.owner = True
                st.success("Access Granted")
                st.rerun()
            else:
                st.error("Wrong Passcode")

    else:
        st.sidebar.success("🧑‍💼 Owner Mode")
        if st.sidebar.button("Logout"):
            st.session_state.owner = False
            st.rerun()

        df_menu = get_menu()

        tab1, tab2 = st.tabs(["📊 Analytics", "⚙️ Manage Menu"])

 
 
        with tab1:
            conn = get_db()
            df = pd.read_sql("SELECT * FROM reviews", conn)

            if df.empty:
                st.warning("No data yet")
            else:
                st.dataframe(df)

                col1, col2 = st.columns(2)
                
                
                with col1:
                    st.subheader("Sentiment")
                    fig, ax = plt.subplots()

                    
                    df['sentiment'] = df['sentiment'].map({
                          b'\x01\x00\x00\x00\x00\x00\x00\x00': 'Positive',
                          b'\x00\x00\x00\x00\x00\x00\x00\x00': 'Negative'
                    })

                    
                    sentiment_counts = df['sentiment'].value_counts()

                    
                    sentiment_counts.plot.pie(
                        autopct='%1.1f%%',
                        labels=sentiment_counts.index,  
                        ax=ax
                    )

                    ax.set_ylabel('')  
                    st.pyplot(fig)
                    

                with col2:
                    st.subheader("Ratings")
                    st.bar_chart(df.groupby("food")["rating"].mean())

                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['date'] = df['timestamp'].dt.date

                st.subheader("📈 Trends")
                st.line_chart(df.groupby("date").size())

                # AI INSIGHT
                st.subheader("🧠 Top Selling Item")
                top_item = df['food'].value_counts().idxmax()
                st.success(f"🔥 {top_item}")


        with tab2:

            st.markdown("### ➕ Add Item")
            new_item = st.text_input("Item Name")
            new_cat = st.selectbox("Category", ["Meals","Drinks","Desserts"])

            if st.button("Add Item"):
                try:
                    conn = get_db()
                    c = conn.cursor()
                    c.execute("INSERT INTO menu VALUES (?,?)", (new_item, new_cat))
                    conn.commit()
                    conn.close()
                    st.success("Added")
                    st.rerun()
                except:
                    st.error("Already exists")

            st.markdown("### ✏️ Edit Item")
            item = st.selectbox("Select Item", df_menu["item"])
            new_name = st.text_input("New Name")
            new_category = st.selectbox("New Category", ["Meals","Drinks","Desserts"])

            if st.button("Update"):
                conn = get_db()
                c = conn.cursor()
                c.execute("UPDATE menu SET item=?, category=? WHERE item=?",
                          (new_name, new_category, item))
                c.execute("UPDATE reviews SET food=? WHERE food=?",
                          (new_name, item))
                conn.commit()
                conn.close()
                st.success("Updated")
                st.rerun()

            st.markdown("### ❌ Delete Item")
            del_item = st.selectbox("Delete Item", df_menu["item"])

            if st.button("Delete"):
                conn = get_db()
                c = conn.cursor()
                c.execute("DELETE FROM menu WHERE item=?", (del_item,))
                c.execute("DELETE FROM reviews WHERE food=?", (del_item,))
                conn.commit()
                conn.close()
                st.success("Deleted")
                st.rerun()