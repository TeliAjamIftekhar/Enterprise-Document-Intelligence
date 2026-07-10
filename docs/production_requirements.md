## Grade 9 Multi-Textbook Scope

The production MVP will support all available Grade 9 textbooks.

### Initial Targets

- Standard: Grade 9
- Maximum initial textbooks: 20
- Supported subjects: English, Marathi, Hindi, Mathematics, Science, History, Geography, and other Grade 9 subjects
- Supported languages: English, Marathi, and Hindi
- Search scope: One selected textbook per question
- Vector architecture: Separate FAISS index for each textbook
- Cross-book search: Not included in the initial MVP

### Required Textbook Metadata

Each textbook must include:

- book_id
- title
- grade
- subject
- language
- board
- academic_year
- version
- status

### Required Chunk Metadata

Each chunk must include:

- chunk_id
- book_id
- grade
- subject
- language
- source
- page
- chapter_id
- chapter_title
- version
- text

### Runtime Behaviour

- The user must select a textbook before asking a question.
- The API must validate the selected book_id.
- Retrieval must run only against the selected textbook.
- Citations must belong to the selected textbook.
- The Lambda runtime should load only the selected textbook index.