# Academic Course Information RAG System

## Overview

This project implements a specialized Retrieval-Augmented Generation (RAG) system designed to process and provide information about academic courses, prerequisites, and faculty information. Built upon the foundations of [MiniRAG](https://github.com/HKUDS/MiniRAG) and [RAGAS](https://github.com/explodinggradients/ragas), this system enhances document processing capabilities with specialized features for academic course information.

## Features

- **Enhanced Document Processing**
  - Specialized table reading for course information
  - Smart chunking system for academic documents
  - Support for Hebrew text processing
  - Custom data splitting for optimal information retrieval

- **Chat Interface**
  - Simple, intuitive chat interface
  - Clean text input/output interface

- **Data Processing Pipeline**
  - PDF to text conversion
  - Document chunking
  - Integration with MiniRAG's RAG capabilities

## Project Structure

```
.
├── chat_interface/       # Web-based chat interface
│   ├── api/             # Backend server
│   └── ui/              # Frontend implementation
├── MiniRAG/             # Core RAG implementation
├── Processed Data/      # Processed academic documents
└── Unprocessed Data/    # Raw PDF documents
```


## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/Horef/dal_final.git
cd dal_final
```

2. Install dependencies:
```bash
pip install -r MiniRAG/requirements.txt
cd chat_interface/ui
npm install
```

3. Process documents:
```bash
python text_from_pdf.py
python data_splitter.py
```

4. Start the interface:
```bash
# In chat_interface/api
python server_minirag.py
# In chat_interface/ui
npm run dev
```

## Usage

1. Place academic PDF documents in the `Unprocessed Data` folder
2. Run the processing scripts to convert and chunk the documents
3. Start the chat interface
4. Ask questions about courses, prerequisites, or faculty information

## Contributing

This project is actively under development. Key areas for contribution include:
- Improving table reading accuracy
- Enhancing chunking algorithms
- Adding support for additional faculties
- Implementing advanced RAG features
- Improving Hebrew language support

