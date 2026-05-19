# %%
from db_utils import HumanMessage, AIMessage, SystemMessage, insert_message, get_current_q_index, increment_q_index
from pydantic_models import QueryInput, QueryResponse, Optional
from langchain_openai import ChatOpenAI
from pydantic_models import QueryInput, QueryResponse

PREDEFINED_QNA = [
    {
        "question": "👋 Hallo! Ich bin dein KI-Assistent. Bevor ich dir weiterhelfen kann, möchte ich dir ein paar Fragen stellen:\nFür welche Gewerke interessierst du dich?",
        "answer": [
            "Ausbau",
            "Sanitär-/Versorgungstechnik",
            "Holzhandwerk",
            "Gartenbau- und Landwirtschaftsbranche",
            "Fahrzeug-/Maschinentechnik",
            "Gesundheitsbranche",
            "Maschinentechnik"
        ],
        "label": "Gewerke ist"
    },
    {
        "question": "In welcher Nutzungsphase befindet sich der Anwendungsfall?",
        "answer": ["Planung & Entwurf", "Bauausführung", "Betrieb & Wartung", "Sanierung & Rückbau"],
        "label": "Nutzungsphase ist"
    },
    {
        "question": "Welche Funktionen sind dir besonders wichtig?",
        "answer": ["Effizienz", "Benutzerfreundlichkeit", "Kostenersparnis"],
        "label": "Besonders wichtig ist"
    }
]

def process_free_text_answer(user_input: str, model, qa) -> Optional[str]:
    # model = "gpt-4o"
    llm = ChatOpenAI(model=model)
    # print("model: ", model)
    prompt = f"""
You are a helpful assistant. The question is: "{qa["question"]}", and the user's answer is: "{user_input}".

Instructions:
- First, correct any spelling or grammar mistakes in the user's input. It's important to keep the same language.
- Then, determine if the corrected input matches the question in any sense.
- If it matches or is clearly related, return the corrected input exactly, without any additional explanation.
- If not, return exactly the word: None.

"""
# - If not, return exactly the word: None, and say why it doesn't match.
# Here are examples of logical answers to the question. This is for inspiration and not an explicit answer template:
# Question:{qa["question"]}
# List of possible answers: {qa["answer"]} 
# """

    messages = [
        SystemMessage(content="You are a language and reasoning assistant."),
        HumanMessage(content=prompt)
    ]

    try:
        # print("+++ Prompt: ", prompt)

        response = llm.invoke(messages).content.strip()
        # print(f"+++ Raw LLM response: '{response}'")

        return None if response == "None" else response
    except Exception as e:
        # print(f"Error during LLM processing: {e}")
        return None


def static_qna_handler(input, model, session_id) -> QueryResponse:

    q_index = get_current_q_index(session_id) # 3
    
    if q_index == 0:
        first_q = PREDEFINED_QNA[0]
        question_text = first_q["question"]
        options_text = "\n".join([f"{i + 1} {opt}" for i, opt in enumerate(first_q["answer"])])
        full_prompt = f"{question_text}\n\n{options_text}"
        insert_message(session_id, HumanMessage(content= input), model)  
        insert_message(session_id, AIMessage(f"{question_text}"), model)
        increment_q_index(session_id)
        return QueryResponse(
            response=full_prompt,
            session_id=session_id,
            model=model,
            is_done=False,
            static_qa=first_q,
        )
    
    elif q_index <= len(PREDEFINED_QNA): # 2 < 3
        previous_q = PREDEFINED_QNA[q_index - 1] # qn = 1
        input_text = input.strip()
        selected_option = None

        # print("### input_text:", input_text, type(input_text))
        try:
            selected_index = int(input_text) - 1
            selected_option = previous_q["answer"][selected_index]

        except Exception:

            try:
                # print("### input_text", input_text)
                selected_option = process_free_text_answer(input_text, model, previous_q)
                # print("### selected_option", selected_option)
                if selected_option is None: 
                    raise ValueError("LLM could not validate the input.")
                
            except Exception as e:
                # Something went wrong (invalid number or LLM failed)
                question_text = previous_q["question"]
                options_text = "\n".join([f"{i + 1} {opt}" for i, opt in enumerate(previous_q["answer"])])
                full_prompt = f"(!) Ungültige Eingabe. Bitte wähle eine gültige Zahl oder gib eine sinnvolle Antwort ein."
                previous_q["question"] = full_prompt
                return QueryResponse(
                    response=full_prompt,
                    session_id=session_id,
                    model=model,
                    is_done=False,
                    static_qa=previous_q
                )
        
        # Record valid answer
        insert_message(session_id, HumanMessage(content=previous_q["label"] + " " + selected_option), model)  
        increment_q_index(session_id) # 

        # If there are more questions
        if q_index < len(PREDEFINED_QNA):
            next_q = PREDEFINED_QNA[q_index] # 2
            question_text = next_q["question"]
            options_text = "\n".join([f"{i + 1} {opt}" for i, opt in enumerate(next_q["answer"])])
            full_prompt = f"{question_text}\n\n{options_text}"
            insert_message(session_id, AIMessage(f"{question_text}"), model)

            return QueryResponse(
                response=full_prompt,
                session_id=session_id,
                model=model,
                is_done=False,
                static_qa=next_q
            )
        
    final_msg = "Danke! Du hast alle Fragen beantwortet.\n\nIch bin jetzt bereit, deine Fragen zu beantworten."
    insert_message(session_id, AIMessage(content=final_msg), model)

    return QueryResponse(
        response=final_msg,
        session_id=session_id,
        model=model,
        is_done=True,
        static_qa=None
    )

