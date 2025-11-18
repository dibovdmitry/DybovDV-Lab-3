from neo4j import GraphDatabase 
import pandas as pd 
from typing import Dict 

# ======================================================================
# НАСТРОЙКИ ПОДКЛЮЧЕНИЯ
# ======================================================================

URI = "bolt://localhost:7687" 
# ВНИМАНИЕ: Замените "password123" на актуальный пароль вашего контейнера Neo4j
AUTH = ("neo4j", "password123") 
driver = GraphDatabase.driver(URI, auth=AUTH)

def test_connection(): 
    with driver.session() as session: 
        result = session.run("RETURN 'Connected to Neo4j' AS message") 
        return result.single()["message"] 

print(test_connection())

# ----------------------------------------------------------------------
# 1. Создание графа знаний
# ----------------------------------------------------------------------

def create_knowledge_graph(tx): 
    # Очистка базы данных 
    tx.run("MATCH (n) DETACH DELETE n") 
    # Создание узлов (сущностей) 
    query = """ 
    CREATE  
    (ai:Domain {name: 'Artificial Intelligence'}), 
    (ml:Technology {name: 'Machine Learning', type: 'ML'}), 
    (dl:Technology {name: 'Deep Learning', type: 'DL'}), 
    (nlp:Technology {name: 'NLP', type: 'NLP'}), 
    (bert:Model {name: 'BERT', developer: 'Google'}), 
    (gpt:Model {name: 'GPT', developer: 'OpenAI'}), 
    (python:Language {name: 'Python', paradigm: 'multi-paradigm'}), 
    (pytorch:Framework {name: 'PyTorch', language: 'Python'}), 
    (tf:Framework {name: 'TensorFlow', language: 'Python'}), 
    // Создание связей 
    (ai)-[:INCLUDES]->(ml), 
    (ai)-[:INCLUDES]->(nlp), 
    (ml)-[:CONTAINS]->(dl), 
    (nlp)-[:USES]->(bert), 
    (nlp)-[:USES]->(gpt), 
    (ml)-[:IMPLEMENTED_IN]->(python), 
    (dl)-[:FRAMEWORK]->(pytorch), 
    (dl)-[:FRAMEWORK]->(tf), 
    (bert)-[:BUILT_WITH]->(tf), 
    (gpt)-[:BUILT_WITH]->(pytorch), 
    (python)-[:HAS_FRAMEWORK]->(pytorch), 
    (python)-[:HAS_FRAMEWORK]->(tf) 
    """ 
    tx.run(query) 

with driver.session() as session: 
    session.execute_write(create_knowledge_graph) 
    print("База знаний создана")

# ----------------------------------------------------------------------
# 2. Функции для чтения и анализа графа
# ----------------------------------------------------------------------

def get_all_nodes(tx): 
    result = tx.run("MATCH (n) RETURN n.name AS name, labels(n) AS labels") 
    return [{"name": record["name"], "labels": record["labels"]} for record in result] 

print("\nВсе узлы в базе:") 
with driver.session() as session:
    nodes = session.execute_read(get_all_nodes) 
for node in nodes: 
    print(f"{node['name']} - {node['labels']}")


def get_relationships(tx): 
    query = """ 
    MATCH (a)-[r]->(b) 
    RETURN a.name AS source, type(r) AS relationship, b.name AS target 
    """ 
    result = tx.run(query) 
    return [{"source": record["source"], "relationship": 
    record["relationship"], "target": record["target"]} for record in result] 

print("\nСвязи в графе:") 
with driver.session() as session:
    relationships = session.execute_read(get_relationships) 
for rel in relationships: 
    print(f"{rel['source']} --{rel['relationship']}--> {rel['target']}")


def find_paths(tx, start_node, end_node): 
    query = """ 
    MATCH path = (a {name: $start_node})-[*]->(b {name: $end_node}) 
    RETURN [node in nodes(path) | node.name] AS path_nodes, 
           [rel in relationships(path) | type(rel)] AS relationships 
    """ 
    result = tx.run(query, start_node=start_node, end_node=end_node) 
    return [{"nodes": record["path_nodes"], "relationships": 
    record["relationships"]} for record in result] 

print("\nПути от AI до BERT:") 
with driver.session() as session:
    paths = session.execute_read(find_paths, "Artificial Intelligence", "BERT") 
for path in paths: 
    print(f"Путь: {' -> '.join(path['nodes'])}")


def analyze_connectivity(tx): 
    # ИСПРАВЛЕНИЕ: Заменен устаревший size((n)--()) на современный COUNT { (n)--() }
    query = """ 
    MATCH (n) 
    RETURN n.name AS node,  
           COUNT { (n)--() } AS degree, 
           labels(n)[0] AS type 
    ORDER BY degree DESC 
    """ 
    result = tx.run(query) 
    return [{"node": record["node"], "degree": record["degree"], "type": 
    record["type"]} for record in result] 

print("\nАнализ связности узлов:") 
with driver.session() as session:
    connectivity = session.execute_read(analyze_connectivity) 
for node in connectivity: 
    print(f"{node['node']} ({node['type']}): {node['degree']} связей")

# ----------------------------------------------------------------------
# 3. Импорт данных из CSV и создание связей
# ----------------------------------------------------------------------

def import_ai_researchers(tx): 
    query = """ 
    LOAD CSV WITH HEADERS FROM 'file:///ai_researchers.csv' AS row 
    CREATE (:Researcher { 
        name: row.name, 
        affiliation: row.affiliation, 
        field: row.field, 
        h_index: toInteger(row.h_index) 
    }) 
    """ 
    tx.run(query) 
    
# Создание CSV файла для импорта
researchers_data = """name,affiliation,field,h_index 
Yann LeCun,Facebook AI Research,Computer Vision,180 
Andrew Ng,Stanford University,Machine Learning,150 
Yoshua Bengio,MILA,Deep Learning,170 
Geoffrey Hinton,University of Toronto,Neural Networks,200 
Demis Hassabis,Google DeepMind,Reinforcement Learning,80 
""" 
with open("neo4j/import/ai_researchers.csv", "w") as f: 
    f.write(researchers_data)

with driver.session() as session:
    session.execute_write(import_ai_researchers) 
    print("Данные исследователей импортированы")


def connect_researchers_to_domains(tx): 
    # ИСПРАВЛЕНИЕ: Добавлены WITH 1 для разделения операторов CREATE и MATCH
    query = """ 
    // 1. Связать всех исследователей с доменом AI
    MATCH (r:Researcher), (d:Domain {name: 'Artificial Intelligence'}) 
    CREATE (r)-[:WORKS_IN]->(d) 
    
    WITH 1 AS dummy
    
    // 2. Связать Яна ЛеКуна с Deep Learning
    MATCH (r:Researcher {name: 'Yann LeCun'}), (dl:Technology {name: 'Deep Learning'}) 
    CREATE (r)-[:CONTRIBUTED_TO]->(dl) 
    
    WITH 1 AS dummy2
    
    // 3. Связать Эндрю Нг с Machine Learning
    MATCH (r:Researcher {name: 'Andrew Ng'}), (ml:Technology {name: 'Machine Learning'}) 
    CREATE (r)-[:CONTRIBUTED_TO]->(ml) 
    """ 
    tx.run(query) 

with driver.session() as session: 
    session.execute_write(connect_researchers_to_domains) 
    print("Связи исследователей созданы")

# ----------------------------------------------------------------------
# 4. Экспорт данных и закрытие соединения
# ----------------------------------------------------------------------

# Функция для экспорта данных (используем tx)
def export_graph_data_tx(tx): 
    query = """ 
    MATCH (n) 
    RETURN n.name AS name,  
           labels(n) AS labels,  
           properties(n) AS properties 
    """ 
    result = tx.run(query) 
    df = pd.DataFrame([dict(record) for record in result]) 
    df.to_csv("graph_export.csv", index=False) 
    return df 

print("\nЭкспортированные данные:")
# Вызов функции через execute_read
with driver.session() as session:
    graph_data = session.execute_read(export_graph_data_tx)
print(graph_data.head())

# Закрытие драйвера
driver.close() 
print("Соединение с Neo4j закрыто")
