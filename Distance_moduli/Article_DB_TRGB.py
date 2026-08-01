# Requiere: streamlit, sqlite3
# Ejecuta con: streamlit run nombre_del_archivo.py

import sqlite3
import streamlit as st
import re
import pandas as pd

df = pd.read_csv('TRGB_general.csv')
df.index.name = 'id'
df.to_sql('TRGB_data', sqlite3.connect('TRGB_database.db'), index=True, if_exists='replace')

# Inicializar la base de datos
def init_db():
    conn = sqlite3.connect("TRGB_database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS TRGB_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        galaxy TEXT,
        bibcode TEXT,
        citation TEXT,
        band TEXT,
        modulus REAL,
        random_error REAL,
        rank INTEGER,
        year INTEGER,
        date_rev INTEGER,
        comments TEXT,
        author_rev TEXT,
        ads_date TEXT,
        julian_date REAL,
        UNIQUE(galaxy, bibcode, modulus, random_error) -- Evita que se repita la combinación de estos 3 campos
    )
    """)



    conn.commit()
    conn.close()



# Agregar medicion de TRGB a la base de datos
def agregar_medicion(datos):
    conn = sqlite3.connect("TRGB_database.db")
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO TRGB_data (galaxy, bibcode, citation, band, modulus, random_error, rank, year, comments, author_rev, date_rev, ads_date, julian_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, datos)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Este error se dispara si se viola la restricción UNIQUE
        return False
    finally:
        conn.close()



# Obtener todos las mediciones de TRGB de la base de datos
def obtener_todas_las_mediciones():
    conn = sqlite3.connect("TRGB_database.db")
    df = pd.read_sql_query("SELECT id, galaxy, bibcode, citation, band, modulus, random_error, rank, year, comments, author_rev, date_rev, ads_date, julian_date FROM TRGB_data", conn)
    conn.close()
    return df

# Eliminar medicion por ID
def eliminar_medicion(medicion_id):
    conn = sqlite3.connect("TRGB_database.db")
    c = conn.cursor()
    c.execute("DELETE FROM TRGB_data WHERE id = ?", (medicion_id,))
    conn.commit()
    conn.close()







# Streamlit UI
st.title("🎇 Compilation of distance measurements using the Tip of the Red Giant Branch (TRGB) method")

init_db()

menu = st.sidebar.selectbox("Menu", [
    "Add record", 
    "Query by galaxy", 
    "Query by bibcode", 
    #"Query by citation",
    #"Query by year",
    #"Query by band", 
    "Show all records",
    "Delete record"
])



if menu == "Add record":
    with st.form("agregar_medicion"):
        galaxy = st.text_input("Galaxy host")
        bibcode = st.text_input("Bibcode")
        citation = st.text_input("Citation")
        band = st.text_input("Photometric band")
        modulus = st.number_input("mu_0", min_value=0.0, max_value=50.0, step=0.001)
        random_error = st.number_input("Random error", min_value=0.0, max_value=50.0, step=0.001)
        rank = st.number_input("Observations origin", min_value=1, max_value=3, step=1)
        year = st.number_input("Publication year", min_value=1950, max_value=2030, step=1)
        date_rev = st.number_input("Date of review", min_value=2022, max_value=2030, step=1)
        comments = st.text_area("Comments")
        author_rev = st.text_input("Reviewer")
        ads_date = st.text_input("ADS date (YYYY-MM-DD)")
        julian_date = st.number_input("ADS Julian date", min_value=0.0, max_value=5000000.0, step=0.1)
        enviar = st.form_submit_button("Guardar")

        if enviar:
            datos = (galaxy, bibcode, citation, band, modulus, random_error, rank, year, comments, author_rev, date_rev, ads_date, julian_date)
            # Intentar guardar y verificar si fue exitoso
            if agregar_medicion(datos):
                st.success("✅ Record added successfully")
            else:
                st.error("⚠️ This record already exists (Same Galaxy, Bibcode, Modulus, Random Error).")




elif menu == "Query by galaxy":
    nombre_galaxia = st.text_input("Introduce the name of the galaxy:")
    if nombre_galaxia:
        conn = sqlite3.connect("TRGB_database.db")
        query = f"""
            SELECT id, galaxy, bibcode, citation, band, modulus, random_error, rank, year, comments, author_rev, date_rev, ads_date, julian_date
            FROM TRGB_data
            WHERE galaxy LIKE ?
        """
        df = pd.read_sql_query(query, conn, params=[f"%{nombre_galaxia}%"])
        conn.close()

        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("No measurements found for that galaxy.")



elif menu == "Query by bibcode":
    bibcode = st.text_input("Introduce the bibcode:")
    if bibcode:
        conn = sqlite3.connect("TRGB_database.db")
        query = f"""
            SELECT id, galaxy, bibcode, citation, band, modulus, random_error, rank, year, comments, author_rev, date_rev, ads_date, julian_date
            FROM TRGB_data
            WHERE bibcode LIKE ?
        """
        df = pd.read_sql_query(query, conn, params=[f"%{bibcode}%"])
        conn.close()

        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("No measurements found for that bibcode.")







elif menu == "Show all records":
    df = obtener_todas_las_mediciones()
    st.subheader("📋 All records in the database")
    st.dataframe(df)

elif menu == "Delete record":
    df = obtener_todas_las_mediciones()
    # CORRECCIÓN: Se limpió el texto del Subheader para quitar el código extraño
    st.subheader("🗑️ Delete record")
    st.dataframe(df)
    
    medicion_id = st.number_input("Enter the ID of the measurement to be deleted:", min_value=1, step=1)
    
    if st.button("Delete"):
        eliminar_medicion(medicion_id)
        st.success("✅ Record deleted successfully")
        # CORRECCIÓN: Recarga la app para limpiar el registro eliminado de la tabla visual de inmediato
        st.rerun()

