Our project is based on the following two github repositories:
- [MiniRAG](https://github.com/HKUDS/MiniRAG)
- [RAGAS](https://github.com/explodinggradients/ragas)

TODO:
- [x] Fix table reading (or at least parse a clear semester - course - number of points relations)
  - sortof done, but not perfect
- [x] Create a data splitter which would produce relevant data chunks from a txt document (i.e., division at each sub-sub-section, or sub-section, etc.)
  - sortof done, but not perfect
- [ ] Integrate into existing MiniRAG (replace their chanker with ours)
- [ ] Maybe (?) add external data source to find course prerequisites, etc..
  -[ ] Download the rest of the faculties - parse the rest.
- [ ] Figure out how to use MiniRAG
    - WIP
- [x] Baseline check 
  - [x] check if exists in MiniRAG, if not - add naive RAG (exists in minirag)

- [ ] Create questions (generate in hebrew) - csv file, columns: Question,Gold Answer,Evidence,Type
 
- [x] Start working on the Interface
  - chatbot - input text box, output text box. Not a dialogue, one shot questions.
  - [x] created a basic chat interface in a corresponding folder
- [ ] Start looking through RAGAS

