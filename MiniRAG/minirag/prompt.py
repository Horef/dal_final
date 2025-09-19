GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = [
    "institution",
    "academic_unit",
    "person_role",
    "academic_program",
    "course",
    "regulation",
    "requirement",
    "academic_concept",
    "document",
    "exam"
]


PROMPTS["entity_extraction"] = """-Goal-
Given a text from a university catalogue and a list of entity types, identify all relevant entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities from the text. For each identified entity, extract the following information:
- entity_name: The official name of the entity, use same language as input text.
- entity_type: One of the following types: [{entity_types}]
- entity_description: A comprehensive description of the entity's function, rules, and attributes as described in the text.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1.
- target_entity: name of the target entity, as identified in step 1.
- relationship_description: explanation of how the source and target entities are related according to the text (e.g., one entity governs another, one is a prerequisite for another).
- relationship_strength: a numeric score from 1-10 indicating the strength of the relationship between the source entity and target entity.
- relationship_keywords: one or more high-level keywords that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details (e.g., 'governance', 'prerequisite', 'requirement', etc).
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level keywords that summarize the main topics of the text, such as 'admissions', 'curriculum', 'academic standing'. These should capture the overarching ideas present in the document.
Format the content-level keywords as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return the output in English as a single list of all entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [institution, person_role, academic_concept, regulation]
Text:
The Technion Senate is the authority that determines all academic matters of the institution in accordance with the constitution and bylaws. The Dean of Undergraduate studies is elected by the full-time tenured professors within the Senate and is entrusted by the Senate to oversee undergraduate studies at the Technion. Students who excel in their studies will be awarded the "Dean's Excellence Award" or the "President's Excellence Award".
################
Output:
("entity"{tuple_delimiter}"Technion Senate"{tuple_delimiter}"institution"{tuple_delimiter}"The Technion Senate is the primary academic authority of the institution, responsible for determining all academic matters, including curricula, degrees, and regulations."){record_delimiter}
("entity"{tuple_delimiter}"Dean of Undergraduate Studies"{tuple_delimiter}"person_role"{tuple_delimiter}"The Dean of Undergraduate Studies is a role elected by professors within the Senate, responsible for overseeing all undergraduate studies at the Technion."){record_delimiter}
("entity"{tuple_delimiter}"Dean's Excellence Award"{tuple_delimiter}"academic_concept"{tuple_delimiter}"The Dean's Excellence Award is an award given to students who demonstrate excellent academic performance."){record_delimiter}
("relationship"{tuple_delimiter}"Technion Senate"{tuple_delimiter}"Dean of Undergraduate Studies"{tuple_delimiter}"The Dean of Undergraduate Studies is elected by and entrusted by the Technion Senate, indicating a hierarchical relationship of governance and oversight."{tuple_delimiter}"governance, authority, election"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Dean of Undergraduate Studies"{tuple_delimiter}"Dean's Excellence Award"{tuple_delimiter}"The Dean's Excellence Award is named after the Dean, who is responsible for academic matters including recognizing student excellence."{tuple_delimiter}"awards, academic excellence"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"academic governance, undergraduate studies, student awards"){completion_delimiter}
#############################
Example 2:

Entity_types: [course, requirement, academic_concept]
Text:
Prior to enrolling in a course, students must pass all the prerequisites. A prerequisite is a course that must be completed (and passed) prior to enrolling in the course in question. Students who successfully pass a course, will receive the allotted number of credits for that course. The recommended schedule of course is based on an average rate of progression of approximately 20 credits per semester.
#############
Output:
("entity"{tuple_delimiter}"Prerequisite"{tuple_delimiter}"requirement"{tuple_delimiter}"A prerequisite is a course that must be successfully completed before a student can enroll in a more advanced, subsequent course."){record_delimiter}
("entity"{tuple_delimiter}"Credits"{tuple_delimiter}"academic_concept"{tuple_delimiter}"Credits are points awarded to students upon successful completion of a course. A certain number of credits is required to obtain a degree, with a recommended progression of 20 credits per semester."){record_delimiter}
("relationship"{tuple_delimiter}"Prerequisite"{tuple_delimiter}"Course"{tuple_delimiter}"A prerequisite course must be passed before a student can enroll in a target course."{tuple_delimiter}"enrollment requirement, course sequence"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Credits"{tuple_delimiter}"Course"{tuple_delimiter}"Successfully passing a course results in the student earning a specific number of credits."{tuple_delimiter}"academic reward, course completion"{tuple_delimiter}8){record_delimiter}
("content_keywords"{tuple_delimiter}"course enrollment, prerequisites, academic credits, study progression"){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""


PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""


PROMPTS[
    "entiti_continue_extraction_mini"
] = """MANY entities were missed in the last extraction.
After summarizing with all the information previously extracted, compared to the original text, it was noticed that the following information was mainly omitted:
{omit}

The types of entities that need to be added can be obtained from Entity_types,
or you can add them yourself.

Entity_types: {entity_types}


Add them below using the same format:
"""


PROMPTS["minirag_query2kwd"] = """---Role---
You are a helpful assistant tasked with identifying answer-type keywords and specific entities in a user's query about a university catalogue.

---Goal---
Given the query, list both the answer-type keywords and the low-level keywords (entities).
-   answer_type_keywords focus on the category of the answer being sought. They must be selected from the Answer type pool.
-   entities_from_query are the specific terms, names, or concepts mentioned in the query.

---Instructions---
- Output the keywords in JSON format.
- The JSON should have two keys: "answer_type_keywords" for the types of the answer and "entities_from_query" for specific entities or details.
- List no more than 3 answer_type_keywords, with the most likely one first.

######################
-Examples-
######################
Example 1:

Query: "How many credits are needed to get a 4-year Bachelor of Science degree?"
Answer type pool: {{
 'REGULATION_PROCEDURE': ['APPEALS PROCESS', 'EXAM RULES'],
 'ADMISSION_REQUIREMENT': ['PSYCHOMETRIC SCORE', 'BAGRUT CERTIFICATE'],
 'CURRICULUM_DETAILS': ['MANDATORY COURSES', 'ELECTIVE CREDITS'],
 'ACADEMIC_STANDING': ['DEANS LIST', 'GPA CALCULATION'],
 'CONTACT_PERSON_DEPARTMENT': ['DEAN OF STUDENTS', 'ACADEMIC SECRETARY']
}}
################
Output:
{{
  "answer_type_keywords": ["CURRICULUM_DETAILS", "REGULATION_PROCEDURE"],
  "entities_from_query": ["credits", "4-year degree", "Bachelor of Science"]
}}
#############################
Example 2:

Query: "What happens if my cumulative GPA falls below 65?"
Answer type pool: {{
 'REGULATION_PROCEDURE': ['APPEALS PROCESS', 'EXAM RULES'],
 'ADMISSION_REQUIREMENT': ['PSYCHOMETRIC SCORE', 'BAGRUT CERTIFICATE'],
 'CURRICULUM_DETAILS': ['MANDATORY COURSES', 'ELECTIVE CREDITS'],
 'ACADEMIC_STANDING': ['DEANS LIST', 'POOR ACADEMIC STANDING'],
 'CONTACT_PERSON_DEPARTMENT': ['DEAN OF STUDENTS', 'ACADEMIC SECRETARY']
}}
################
Output:
{{
  "answer_type_keywords": ["ACADEMIC_STANDING", "REGULATION_PROCEDURE"],
  "entities_from_query": ["cumulative GPA", "GPA below 65", "poor academic standing"]
}}
#############################
Example 3:

Query: "Who do I contact to defer my admission for a semester?"
Answer type pool: {{
 'REGULATION_PROCEDURE': ['APPEALS PROCESS', 'DEFERRING ADMISSION'],
 'ADMISSION_REQUIREMENT': ['PSYCHOMETRIC SCORE', 'BAGRUT CERTIFICATE'],
 'CURRICULUM_DETAILS': ['MANDATORY COURSES', 'ELECTIVE CREDITS'],
 'ACADEMIC_STANDING': ['DEANS LIST', 'GPA CALCULATION'],
 'CONTACT_PERSON_DEPARTMENT': ['DEAN OF UNDERGRADUATE STUDIES', 'ACADEMIC SECRETARY']
}}
################
Output:
{{
  "answer_type_keywords": ["CONTACT_PERSON_DEPARTMENT", "REGULATION_PROCEDURE", "ADMISSION_REQUIREMENT"],
  "entities_from_query": ["defer admission", "contact person"]
}}
#############################
-Real Data-
######################
Query: {query}
Answer type pool:{TYPE_POOL}
######################
Output:
"""


PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---
You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query about a university catalogue.

---Goal---
Given the query, list both high-level and low-level keywords.
- High-level keywords focus on overarching academic concepts or themes (e.g., 'admission standards', 'degree requirements').
- Low-level keywords focus on specific entities or details from the query (e.g., 'Physics placement exam', '155 credits').

---Instructions---
- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
Example 1:

Query: "How does the university determine academic honors like summa cum laude?"
################
Output:
{{
  "high_level_keywords": ["Academic honors", "Graduation distinction", "GPA requirements"],
  "low_level_keywords": ["summa cum laude", "cum laude", "academic distinction"]
}}
#############################
Example 2:

Query: "What are the prerequisites for the Advanced Technical English B course?"
################
Output:
{{
  "high_level_keywords": ["Course prerequisites", "Enrollment requirements", "English studies"],
  "low_level_keywords": ["Advanced Technical English B", "prerequisites"]
}}
#############################
Example 3:

Query: "Can I get credit for social activities or reserve duty?"
################
Output:
{{
  "high_level_keywords": ["Academic credits", "Elective credits", "Extracurricular activities"],
  "low_level_keywords": ["social activity", "reserve duty", "credit"]
}}
#############################
-Real Data-
######################
Query: {query}
######################
Output:
"""

PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""