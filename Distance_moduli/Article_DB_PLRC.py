# Requiere: streamlit, sqlite3
# Ejecuta con: streamlit run nombre_del_archivo.py

import sqlite3
import streamlit as st
import re
import pandas as pd

df = pd.read_csv('PLRC_general_v1.csv')
df.index.name = 'id'
df.to_sql('PLRC_data', sqlite3.connect('PLRC_database.db'), index=True, if_exists='replace')

# Inicializar la base de datos
def init_db():
    conn = sqlite3.connect("PLRC_database.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS PLRC_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        galaxy TEXT,
        bibcode TEXT,
        citation TEXT,
        band TEXT,
        modulus REAL,
        random_error REAL,
        systematic_error REAL,
        total_error REAL,
        category_error INTEGER,
        cepheids_number INTEGER,
        metal_correction INTEGER,
        zero_point TEXT,
        probe_quality INTEGER,
        publication_year INTEGER,
        review_year INTEGER,
        comments TEXT,
        review_author TEXT,
        ads_date TEXT,
        ads_jd REAL,
        UNIQUE(galaxy, bibcode, modulus, random_error,total_error ) -- Evita que se repita la combinación de estos 3 campos
    )
    """)



    conn.commit()
    conn.close()



# Agregar medicion de PLRC a la base de datos
def agregar_medicion(datos):
    conn = sqlite3.connect("PLRC_database.db")
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO PLRC_data (galaxy,bibcode,citation,band,modulus,random_error,systematic_error,total_error,category_error,cepheids_number,metal_correction,zero_point,probe_quality,publication_year,review_year,comments,review_author,ads_date,ads_jd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,? ,? ,? ,?)
        """, datos)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Este error se dispara si se viola la restricción UNIQUE
        return False
    finally:
        conn.close()



# Obtener todos las mediciones de PLRC de la base de datos
def obtener_todas_las_mediciones():
    conn = sqlite3.connect("PLRC_database.db")
    df = pd.read_sql_query("SELECT id, galaxy,bibcode,citation,band,modulus,random_error,systematic_error,total_error,category_error,cepheids_number,metal_correction,zero_point,probe_quality,publication_year,review_year,comments,review_author,ads_date,ads_jd FROM PLRC_data", conn)
    conn.close()
    return df

# Eliminar medicion por ID
def eliminar_medicion(medicion_id):
    conn = sqlite3.connect("PLRC_database.db")
    c = conn.cursor()
    c.execute("DELETE FROM PLRC_data WHERE id = ?", (medicion_id,))
    conn.commit()
    conn.close()







# Streamlit UI
st.title("🎇 Compilation of distance measurements using the Period-Luminosity Relation of Cepheids (PLRC) method")

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
        system_error = st.number_input("Systematic error", min_value=0.0, max_value=50.0, step=0.001)
        total_error = st.number_input("Total error", min_value=0.0, max_value=50.0, step=0.001)
        Cate = st.number_input("Category error", min_value=1, max_value=6, step=1)
        N_Ceph = st.number_input("Number of Cepheids", min_value=-1, max_value=1000, step=1)
        metal_cor = st.toggle("Metal correction")
        ZP = st.text_input("Zero Point")
        rank = st.number_input("Observations origin", min_value=1, max_value=3, step=1)
        year = st.number_input("Publication year", min_value=1950, max_value=2030, step=1)
        date_rev = st.number_input("Date of review", min_value=2022, max_value=2030, step=1)
        comments = st.text_area("Comments")
        author_rev = st.text_input("Reviewer")
        ads_date = st.text_input("ADS date (YYYY-MM-DD)")
        julian_date = st.number_input("ADS Julian date", min_value=0.0, max_value=5000000.0, step=0.1)
        enviar = st.form_submit_button("Guardar")

        if enviar:
            datos = (
                    galaxy,         # galaxy
                    bibcode,        # bibcode
                    citation,       # citation
                    band,           # band
                    modulus,        # modulus
                    random_error,   # random_error
                    random_error,
                    system_error,
                    total_error,
                    Cate,
                    N_Ceph,
                    int(metal_cor),
                    ZP,
                    rank,           # probe_quality
                    year,           # publication_year
                    date_rev,       # review_year
                    comments,       # comments
                    author_rev,     # review_author
                    ads_date,       # ads_date
                    julian_date     # ads_jd
                )
            # Intentar guardar y verificar si fue exitoso
            if agregar_medicion(datos):
                st.success("✅ Record added successfully")
            else:
                st.error("⚠️ This record already exists (Same Galaxy, Bibcode, Modulus, Random Error, Total Error).")




elif menu == "Query by galaxy":
    nombre_galaxia = st.text_input("Introduce the name of the galaxy:")
    if nombre_galaxia:
        conn = sqlite3.connect("PLRC_database.db")
        query = f"""
            SELECT id, galaxy,bibcode,citation,band,modulus,random_error,systematic_error,total_error,category_error,cepheids_number,metal_correction,zero_point,probe_quality,publication_year,review_year,comments,review_author,ads_date,ads_jd
            FROM PLRC_data
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
        conn = sqlite3.connect("PLRC_database.db")
        query = f"""
            SELECT id, galaxy,bibcode,citation,band,modulus,random_error,systematic_error,total_error,category_error,cepheids_number,metal_correction,zero_point,probe_quality,publication_year,review_year,comments,review_author,ads_date,ads_jd
            FROM PLRC_data
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

