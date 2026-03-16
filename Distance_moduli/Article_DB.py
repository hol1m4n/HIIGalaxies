# Requiere: streamlit, sqlite3
# Ejecuta con: streamlit run nombre_del_archivo.py

import sqlite3
import streamlit as st
import re
import pandas as pd

# Inicializar la base de datos
def init_db():
    conn = sqlite3.connect("biblioteca.db")
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS libros (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        titulo TEXT,
        autor TEXT,
        tema TEXT,
        anio INTEGER,
        solucionario BOOLEAN,
        ruta TEXT,
        notas TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nombre TEXT UNIQUE
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS libro_tags (
        libro_id INTEGER,
        tag_id INTEGER,
        FOREIGN KEY(libro_id) REFERENCES libros(id),
        FOREIGN KEY(tag_id) REFERENCES tags(id),
        UNIQUE(libro_id, tag_id)
    )
    """)

    conn.commit()
    conn.close()

# Agregar libro
def agregar_libro(datos, tags):
    conn = sqlite3.connect("biblioteca.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO libros (titulo, autor, tema, anio, solucionario, ruta, notas)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, datos)
    libro_id = c.lastrowid

    for tag in tags:
        c.execute("INSERT OR IGNORE INTO tags (nombre) VALUES (?)", (tag,))
        c.execute("SELECT id FROM tags WHERE nombre = ?", (tag,))
        tag_id = c.fetchone()[0]
        c.execute("INSERT OR IGNORE INTO libro_tags (libro_id, tag_id) VALUES (?, ?)", (libro_id, tag_id))

    conn.commit()
    conn.close()

# Buscar libros por tag
def buscar_libros_por_tag(tag):
    conn = sqlite3.connect("biblioteca.db")
    df = pd.read_sql_query("""
        SELECT libros.id, titulo, autor, tema, anio, solucionario, ruta, notas
        FROM libros
        JOIN libro_tags ON libros.id = libro_tags.libro_id
        JOIN tags ON tags.id = libro_tags.tag_id
        WHERE tags.nombre = ?
    """, conn, params=[tag])
    conn.close()
    return df

# Obtener todos los libros
def obtener_todos_los_libros():
    conn = sqlite3.connect("biblioteca.db")
    df = pd.read_sql_query("SELECT id, titulo, autor, tema, anio, solucionario, ruta, notas FROM libros", conn)
    conn.close()
    return df

# Eliminar libro por ID
def eliminar_libro(libro_id):
    conn = sqlite3.connect("biblioteca.db")
    c = conn.cursor()
    c.execute("DELETE FROM libro_tags WHERE libro_id = ?", (libro_id,))
    c.execute("DELETE FROM libros WHERE id = ?", (libro_id,))
    conn.commit()
    conn.close()

# Streamlit UI
st.title("📚 Biblioteca de Libros de Física y Astrofísica")

init_db()

menu = st.sidebar.selectbox("Menú", [
    "Agregar libro", 
    "Buscar por tag", 
    "Buscar por autor", 
    "Buscar por carpeta", 
    "Mostrar todos",
    "Eliminar libro"
])

if menu == "Agregar libro":
    with st.form("agregar_libro"):
        titulo = st.text_input("Título")
        autor = st.text_input("Autor")
        tema = st.text_input("Tema principal")
        anio = st.number_input("Año", min_value=1800, max_value=2100, step=1)
        solucionario = st.checkbox("¿Tiene solucionario?")
        ruta = st.text_input("Ruta del archivo (puede ser enlace de Google Drive compartido)")
        notas = st.text_area("Notas adicionales")
        tags_input = st.text_input("Tags (separados por comas)")
        enviar = st.form_submit_button("Guardar")

        if enviar:
            tags = [t.strip().lower() for t in tags_input.split(",") if t.strip()]
            datos = (titulo, autor, tema, anio, solucionario, ruta, notas)
            agregar_libro(datos, tags)
            st.success("✅ Libro agregado correctamente")

elif menu == "Buscar por tag":
    tag = st.text_input("Introduce un tag (ej. relatividad)")
    if tag:
        df = buscar_libros_por_tag(tag.lower())
        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("No se encontraron libros con ese tag.")

elif menu == "Mostrar todos":
    df = obtener_todos_los_libros()
    st.subheader("📋 Todos los libros en la base de datos")
    st.dataframe(df)

elif menu == "Buscar por autor":
    nombre_autor = st.text_input("Introduce el nombre o parte del nombre del autor:")
    if nombre_autor:
        conn = sqlite3.connect("biblioteca.db")
        query = f"""
            SELECT id, titulo, autor, tema, anio, solucionario, ruta, notas
            FROM libros
            WHERE autor LIKE ?
        """
        df = pd.read_sql_query(query, conn, params=[f"%{nombre_autor}%"])
        conn.close()

        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("No se encontraron libros con ese autor.")

elif menu == "Buscar por carpeta":
    subruta = st.text_input("Introduce el nombre de la carpeta o parte de la ruta:")
    if subruta:
        conn = sqlite3.connect("biblioteca.db")
        query = f"""
            SELECT id, titulo, autor, tema, anio, solucionario, ruta, notas
            FROM libros
            WHERE ruta LIKE ?
        """
        df = pd.read_sql_query(query, conn, params=[f"%{subruta}%"])
        conn.close()

        if not df.empty:
            st.dataframe(df)
        else:
            st.warning("No se encontraron libros en esa ubicación.")

elif menu == "Eliminar libro":
    df = obtener_todos_los_libros()
    st.subheader("🗑️ Eliminar libro")
    st.dataframe(df)
    libro_id = st.number_input("Introduce el ID del libro a eliminar:", min_value=1, step=1)
    if st.button("Eliminar"):
        eliminar_libro(libro_id)
        st.success("✅ Libro eliminado correctamente")
