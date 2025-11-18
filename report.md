# МИНИCTEPCTBO НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ
## Федеральное государственное автономное образовательное учреждение высшего образования «Северо-Кавказский федеральный университет» Институт перспективной инженерии

### Отчет по лабораторной работе 3
### Работа с векторными (ChromaDB) и графовыми (Neo4j) базами данных
Дата: 2025-11-18 \
Семестр: [2 курс 1 полугодие - 3 семестр] \
Группа: ПИН-м-о-24-1 \
Дисциплина: Технологии программирования \
Студент: Дыбов Д.В.
#### Цель работы
Освоение базовых принципов работы с векторными базами данных на примере ChromaDB и графовыми базами данных на примере Neo4j. Получение практических навыков создания коллекций, генерации эмбеддингов, выполнения семантического поиска и интеграции с ML-моделями; создания узлов и связей, выполнения запросов на языке Cypher и визуализации графовых структур.
#### Теоретическая часть
Краткие изученные концепции:
- Векторные базы данных (ChromaDB): коллекции, эмбеддинги, семантический поиск, хранение метаданных.
- Модели эмбеддингов: принцип работы sentence-transformers, важность версионности и нормализации.
- Графовые базы данных (Neo4j): узлы, отношения, свойства, модель данных Property Graph.
- Язык запросов Cypher: MATCH, CREATE, MERGE, RETURN, работу с ограничениями и индексами.
- Интеграция с ML: генерация эмбеддингов, сохранение их в векторной БД, семантический поиск как этап пайплайна.
#### Практическая часть
##### Выполненные задачи
- [x] Установлены пакеты chromadb, sentence-transformers, pandas, numpy.
- [x] Создана рабочая директория и добавлен файл chroma_demo.py для работы с ChromaDB.
- [x] Наполнен файл chroma_demo.py и запущен скрипт; проверен результат семантического поиска.
- [x] Запущен контейнер Neo4j и проверена работоспособность.
- [x] Открыт веб-интерфейс Neo4j по адресу http://localhost:7474; выполнён вход.
- [x] Создан и запущен файл neo4j_demo.py; выполнен запрос MATCH (n) RETURN n LIMIT 25 и произведена визуализация графа.
- [x] Остановлен контейнер Neo4j.

##### Ключевые фрагменты кода
- Скрипт chromo_demo.py:
```python
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from typing import List, Dict

model = SentenceTransformer('all-MiniLM-L6-v2')
print("Модель для эмбеддингов загружена")

client = chromadb.Client()
collection = client.create_collection(
    name="documents_collection",
    metadata={"hnsw:space": "cosine"}  # Использование косинусного расстояния
)
print("Коллекция создана")

documents = [
"Машинное обучение - это область искусственного интеллекта",
"Глубокое обучение использует нейронные сети с множеством слоев",
"Трансформеры revolutionized обработку естественного языка",
"BERT является популярной моделью для понимания текста",
"GPT модели используются для генерации текста",
"Векторные базы данных хранят embeddings для семантического поиска",
"ChromaDB - это open-source векторная база данных",
"Semantic search позволяет находить документы по смыслу",
"Neural networks inspired биологическими нейронными сетями",
"Natural Language Processing обрабатывает человеческий язык"
]
metadata = [{"category": "AI", "source": "educational"} for _ in
range(len(documents))]
ids = [f"doc_{i}" for i in range(len(documents))]
print(f"Подготовлено {len(documents)} документов")

embeddings = model.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadata,
    ids=ids
)
print("Документы добавлены в коллекцию")

def semantic_search(query: str, n_results: int = 3):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "distances", "metadatas"]
    )
    return results

test_queries = [
"искусственный интеллект",
"нейронные сети",
"обработка текста",
"базы данных"
]
print("\nРезультаты семантического поиска:")
for query in test_queries:
    print(f"\nЗапрос: '{query}'")
    results = semantic_search(query)
    for i, (doc, distance) in enumerate(zip(results['documents'][0],
    results['distances'][0])):
        print(f"{i+1}. {doc} (расстояние: {distance:.4f})")

def filtered_search(query: str, filter_dict: Dict, n_results: int = 2):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=filter_dict,
        include=["documents", "distances", "metadatas"]
    )
    return results
    
print("\n\nПоиск с фильтрацией по категории:")
results = filtered_search(
    "модели машинного обучения",
    {"category": "AI"},
    n_results=2
)
for i, (doc, distance) in enumerate(zip(results['documents'][0],
results['distances'][0])):
    print(f"{i+1}. {doc} (расстояние: {distance:.4f})")

print(f"\nИнформация о коллекции:")
print(f"Количество документов: {collection.count()}")
sample_docs = collection.get(ids=["doc_0", "doc_1"])
print("\nПримеры документов:")
for doc in sample_docs['documents']:
    print(f"- {doc}")

persistent_client = chromadb.PersistentClient(path="./chroma_db")
persistent_collection = persistent_client.create_collection(
    name="persistent_docs",
    metadata={"hnsw:space": "cosine"}
)
persistent_collection.add(
    documents=documents[:5],  # Первые 5 документов
    embeddings=embeddings[:5],
    metadatas=metadata[:5],
    ids=ids[:5]
)
print("Персистентная коллекция создана и заполнена")
loaded_collection = persistent_client.get_collection("persistent_docs")
print(f"Загружено документов: {loaded_collection.count()}")

def create_news_collection():
    news_data = [
"Рынок акций вырос на 2% сегодня",
"Новая технология в области искусственного интеллекта представлена",
"Криптовалюты показывают волатильность на рынке",
"Ученые сделали breakthrough в квантовых вычислениях",
"Центральные банки обсуждают monetary policy"
    ]
    news_metadata = [
        {"category": "finance", "date": "2024-01-15"},
        {"category": "technology", "date": "2024-01-15"},
        {"category": "crypto", "date": "2024-01-14"},
        {"category": "science", "date": "2024-01-14"},
        {"category": "economics", "date": "2024-01-13"}
    ]
    news_ids = [f"news_{i}" for i in range(len(news_data))]
    news_embeddings = model.encode(news_data).tolist()
    news_collection = client.create_collection(name="news_collection")
    news_collection.add(
        documents=news_data,
        embeddings=news_embeddings,
        metadatas=news_metadata,
        ids=news_ids
    )
    return news_collection

news_coll = create_news_collection()
results = news_coll.query(
    query_embeddings=model.encode(["финансовые новости"]).tolist(),
    n_results=2
)
print("\nПоиск в новостной коллекции:")
for doc in results['documents'][0]:
    print(f"- {doc}")
```
- Скрипт chromo_demo.py:
```python

```
##### Результаты выполнения
1. Произведена установка зависимостей pip install chromadb sentence-transformers pandas numpy
![скриншот](report/Screenshot1.png "Рисунок") \
Рисунок 1 — Установка пакетов (скриншот).

2. Создана директория проекта; добавлен chroma_demo.py.

Рисунок 2 — Создание рабочей директории.

Реализация chroma_demo.py:

Наполнение скрипта: загрузка текстов, генерация эмбеддингов через sentence-transformers, создание коллекции ChromaDB, вставка объектов, выполнение семантического поиска по запросам.

Рисунок 3 — Фрагменты кода chroma_demo.py.

Запуск chroma_demo.py:

Скрипт выполнен; результаты поиска показали релевантность даже при неточных ключевых словах (примеры: «базы данных», «нейронные сети»).

Рисунок 4 — Результат выполнения кода.

Запуск Neo4j:

Запущен Docker-контейнер Neo4j (порт 7474); проверка работы контейнера.

Рисунок 5 — Запуск Neo4j контейнера.

Подготовка и проверка вспомогательных файлов:

Проверено содержимое fine_turning.py (если использовался) — Рисунок 6.

Работа с Neo4j через веб-интерфейс:

Открыт http://localhost:7474; выполнён вход — Рисунок 7 (страница авторизации), Рисунок 8 (авторизованный вход).

Реализация neo4j_demo.py:

Скрипт создает узлы и связи (например, TECHNOLOGY, RESEARCHER, PAPER, CONTRIBUTED_TO), выполняет вставку данных и простые запросы.

Рисунок 9 — Скрипт neo4j_demo.py; Рисунок 10 — Результат выполнения.

Визуализация графа:

Выполнен запрос MATCH (n) RETURN n LIMIT 25; получены таблицы и визуализация — Рисунок 11 (таблицы), Рисунок 12 (визуализация).

Анализ: граф отображает связи между концепциями ИИ, технологиями и исследователями; отмечена неполнота связей CONTRIBUTED_TO у некоторых исследователей.

Остановка контейнера:

Остановлен Neo4j-контейнер — Рисунок 13.


Результаты выполнения
- Установлены зависимости и подготовлено окружение (скриншоты установки) — Рисунок 1.
- Создано рабочее окружение и файлы проекта — Рисунок 2, Рисунок 3.
- chroma_demo.py выполнил семантический поиск успешно; релевантные результаты для тематических запросов — Рисунок 4.
- Neo4j запущен и доступен через веб-интерфейс; выполнены вставки и запросы, получена визуализация графа — Рисунки 5–12.
- Отмечены недостатки данных: часть исследователей не имеет связи CONTRIBUTED_TO, что снижает полноту графа.
- Контейнер Neo4j остановлен — Рисунок 13.

Тестирование
- [x] Модульные тесты — не применялись.
- [x] Интеграционные тесты — проверены взаимодействия между компонентами (генерация эмбеддингов → запись в ChromaDB; создание/запросы в Neo4j).
- [x] Производительность — не в фокусе данной работы; операции завершились в разумные сроки в тестовой среде.

Выводы
Выполнение лабораторной работы №3 позволило: /
Получить практические навыки работы с векторными БД (ChromaDB): создание коллекций, генерация и запись эмбеддингов, семантический поиск.
Освоить базовые операции с графовой БД (Neo4j): создание узлов и отношений, выполнение запросов на Cypher, визуализацию графовых структур.
Выявить области улучшения: дополнять данные связями CONTRIBUTED_TO, логировать версии моделей эмбеддингов и метаданные коллекций для воспроизводимости.

Приложения и артефакты
chroma_demo.py — скрипт для работы с ChromaDB (в рабочей директории).

neo4j_demo.py — скрипт для работы с Neo4j.

fine_turning.py — вспомогательный файл (при наличии).

Скриншоты и результаты тестов: Рисунок 1..13 (папка report при подготовке отчёта).
