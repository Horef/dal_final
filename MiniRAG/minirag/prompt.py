GRAPH_FIELD_SEP = "<מפריד>"

PROMPTS = {}

PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|שלם|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["סמסטר", "קורס", "ניקוד", "מגמה", "חובה", "בחירה"]


PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_keywords>{tuple_delimiter}<relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"{tuple_delimiter}<high_level_keywords>)

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}"power dynamics, perspective shift"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}"shared goals, rebellion"{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}"conflict resolution, mutual respect"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}"ideological conflict, rebellion"{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}"reverence, technological significance"{tuple_delimiter}9){record_delimiter}
("content_keywords"{tuple_delimiter}"power dynamics, ideological conflict, discovery, rebellion"){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}"decision-making, external influence"{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}"mission evolution, active participation"{tuple_delimiter}9){completion_delimiter}
("content_keywords"{tuple_delimiter}"mission evolution, decision-making, active participation, cosmic significance"){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}"communication, learning process"{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}"leadership, exploration"{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}"collective action, cosmic significance"{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}"power dynamics, autonomy"{tuple_delimiter}7){record_delimiter}
("content_keywords"{tuple_delimiter}"first contact, control, communication, cosmic significance"){completion_delimiter}
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


PROMPTS["minirag_query2kwd"] = """---תפקיד---

אתה עוזר שתפקידו לזהות מילות מפתח מסוג תשובה ומילות מפתח ברמה נמוכה בשאילתת המשתמש.

---מטרה---

בהינתן השאילתה, יש לרשום מילות מפתח מסוג תשובה ומילות מפתח ברמה נמוכה.
מילות מפתח מסוג תשובה מתמקדות בסוג התשובה לשאילתה מסוימת, בעוד שמילות מפתח ברמה נמוכה מתמקדות בישויות, פרטים או מונחים קונקרטיים.
יש לבחור את מילות המפתח מסוג תשובה מתוך מאגר סוגי התשובות.
מאגר זה הוא בצורת מילון, כאשר המפתח מייצג את הסוג שעליך לבחור והערך מייצג דוגמאות.

---הוראות---

- יש להפיק את מילות המפתח בפורמט JSON.
- ה-JSON צריך להכיל שני מפתחות:
  - "answer_type_keywords" עבור סוגי התשובה. ברשימה זו, הסוגים בעלי הסבירות הגבוהה ביותר צריכים להופיע ראשונים. לא יותר מ-3.
  - "entities_from_query" עבור ישויות או פרטים ספציפיים. יש לחלץ אותם מהשאילתה.
######################
-דוגמאות-
######################
דוגמה 1:

שאילתה: "מהן דרישות הקבלה לתואר ראשון בהנדסת חשמל?"
מאגר סוגי תשובות: {{
 'דרישות קבלה': ['סף קבלה', 'תנאי קבלה'],
 'תכנית לימודים': ['נקודות זכות', 'רשימת קורסים'],
 'מידע על קורס': ['שעות שבועיות', 'דרישות קדם'],
 'פקולטה': ['הפקולטה להנדסת חשמל', 'הפקולטה למדעי המחשב'],
 'איש סגל': ['פרופסור', 'מרצה']
}}
################
פלט:
{{
  "answer_type_keywords": ["דרישות קבלה", "תכנית לימודים"],
  "entities_from_query": ["דרישות קבלה", "תואר ראשון", "הנדסת חשמל"]
}}
#############################
דוגמה 2:

שאילתה: "כמה נקודות זכות צריך לצבור בקורס מבוא למדעי המחשב?"
מאגר סוגי תשובות: {{
 'דרישות קבלה': ['סף קבלה', 'תנאי קבלה'],
 'תכנית לימודים': ['נקודות זכות', 'רשימת קורסים'],
 'מידע על קורס': ['שעות שבועיות', 'דרישות קדם'],
 'פקולטה': ['הפקולטה להנדסת חשמל', 'הפקולטה למדעי המחשב'],
 'איש סגל': ['פרופסור', 'מרצה']
}}
################
פלט:
{{
  "answer_type_keywords": ["מידע על קורס", "תכנית לימודים"],
  "entities_from_query": ["נקודות זכות", "מבוא למדעי המחשב"]
}}
#############################
דוגמה 3:

שאילתה: "האם יש קורסי בחירה בפקולטה למדעי הנתונים?"
מאגר סוגי תשובות: {{
 'דרישות קבלה': ['סף קבלה', 'תנאי קבלה'],
 'תכנית לימודים': ['נקודות זכות', 'רשימת קורסים'],
 'מידע על קורס': ['שעות שבועיות', 'דרישות קדם'],
 'פקולטה': ['הפקולטה להנדסת חשמל', 'הפקולטה למדעי המחשב'],
 'איש סגל': ['פרופסור', 'מרצה']
}}
################
פלט:
{{
  "answer_type_keywords": ["תכנית לימודים", "פקולטה"],
  "entities_from_query": ["קורסי בחירה", "הפקולטה למדעי הנתונים"]
}}
#############################
דוגמה 4:

שאילתה: "מי המרצה של הקורס אלגברה לינארית?"
מאגר סוגי תשובות: {{
 'דרישות קבלה': ['סף קבלה', 'תנאי קבלה'],
 'תכנית לימודים': ['נקודות זכות', 'רשימת קורסים'],
 'מידע על קורס': ['שעות שבועיות', 'דרישות קדם'],
 'פקולטה': ['הפקולטה להנדסת חשמל', 'הפקולטה למדעי המחשב'],
 'איש סגל': ['פרופסור', 'מרצה']
}}
################
פלט:
{{
  "answer_type_keywords": ["איש סגל", "מידע על קורס"],
  "entities_from_query": ["מרצה", "אלגברה לינארית"]
}}
#############################

-נתונים אמיתיים-
######################
שאילתה: {query}
מאגר סוגי תשובות:{TYPE_POOL}
######################
פלט:

"""


PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "מצטער, אני לא יכול לספק תשובה לשאלה זו."

PROMPTS["rag_response"] = """---תפקיד---

אתה עוזר המסייע בתשובה לשאלות על נתונים בטבלאות שסופקו.


---מטרה---

צור תגובה באורך ובתבנית היעד המגיבה לשאלת המשתמש, מסכמת את כל המידע בטבלאות הנתונים המתאימות לאורך ולתבנית התגובה, ומשלבת כל ידע כללי רלוונטי.
אם אינך יודע את התשובה, פשוט אמור זאת. אל תמציא שום דבר.
אל תכלול מידע שהראיות התומכות בו אינן מסופקות.

---אורך ותבנית תגובת היעד---

{response_type}

---טבלאות נתונים---

{context_data}

הוסף קטעים והערות לתגובה לפי הצורך לאורך ולתבנית. עצב את התגובה ב-markdown.
"""

PROMPTS["keywords_extraction"] = """---תפקיד---

אתה עוזר שתפקידו לזהות מילות מפתח ברמה גבוהה וברמה נמוכה בשאילתת המשתמש.

---מטרה---

בהינתן השאילתה, יש לרשום מילות מפתח ברמה גבוהה וברמה נמוכה. מילות מפתח ברמה גבוהה מתמקדות במושגים או נושאים כלליים, בעוד שמילות מפתח ברמה נמוכה מתמקדות בישויות, פרטים או מונחים קונקרטיים.

---הוראות---

- יש להפיק את מילות המפתח בפורמט JSON.
- ה-JSON צריך להכיל שני מפתחות:
  - "high_level_keywords" עבור מושגים או נושאים כלליים.
  - "low_level_keywords" עבור ישויות או פרטים ספציפיים.

######################
-דוגמאות-
######################
דוגמה 1:

שאילתה: "מהן דרישות הקבלה לתואר ראשון בהנדסת חשמל?"
################
פלט:
{{
  "high_level_keywords": ["דרישות קבלה", "תואר ראשון", "הנדסת חשמל"],
  "low_level_keywords": ["ציון פסיכומטרי", "בגרות במתמטיקה", "סף קבלה", "תנאי קבלה"]
}}
#############################
דוגמה 2:

שאילתה: "כמה נקודות זכות צריך לצבור בקורס מבוא למדעי המחשב?"
################
פלט:
{{
  "high_level_keywords": ["נקודות זכות", "קורס חובה", "מדעי המחשב"],
  "low_level_keywords": ["מבוא למדעי המחשב", "נקודות זכות", "שעות שבועיות", "דרישות קדם"]
}}
#############################
דוגמה 3:

שאילתה: "האם יש קורסי בחירה בפקולטה למדעי הנתונים?"
################
פלט:
{{
  "high_level_keywords": ["קורסי בחירה", "מדעי הנתונים", "תכנית לימודים"],
  "low_level_keywords": ["רשימת קורסים", "נקודות בחירה", "פקולטה", "קטלוג קורסים"]
}}
#############################
-נתונים אמיתיים-
######################
שאילתה: {query}
######################
פלט:

"""

PROMPTS["naive_rag_response"] = """---תפקיד---

אתה עוזר המסייע בתשובה לשאלות על המסמכים שסופקו.


---מטרה---

צור תגובה באורך ובתבנית היעד המגיבה לשאלת המשתמש, מסכמת את כל המידע בטבלאות הנתונים המתאימות לאורך ולתבנית התגובה, ומשלבת כל ידע כללי רלוונטי.
אם אינך יודע את התשובה, פשוט אמור זאת. אל תמציא שום דבר.
אל תכלול מידע שהראיות התומכות בו אינן מסופקות.

---אורך ותבנית תגובת היעד---

{response_type}

---מסמכים---

{content_data}

הוסף קטעים והערות לתגובה לפי הצורך לאורך ולתבנית. עצב את התגובה ב-markdown.
"""
