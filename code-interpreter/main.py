from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import (
    create_csv_agent,
    create_python_agent,
)


def main():
    print("start...")

    # model="gpt-4" 로 변경하면 훨씬 안정적으로 응답을 생성함
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    python_agent_executor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    csv_agent_executor = create_csv_agent(
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        path="episode_info.csv",
        verbose=True,
    )

    tools = [
        Tool(
            name="python_agent",
            func=python_agent_executor.invoke,
            description="""
                useful when you need to transform natural language and write from it python and execute the python code,
                returning the results of the code execution,
                TIPS: DO NOT SEND PYTHON CODE TO THIS TOOL DIRECTLY.
                """,
        ),
        Tool(
            name="csv_agent",
            func=csv_agent_executor.invoke,
            description="""
                useful when you need to answer question over episode_info.csv file,
                takes an input the entire question and returns the answer after running pandas calculations
                """,
        ),
    ]

    # 일반 도구를 선택하는 prompt는 OpenAI Function Interface로 대체할 수 있음
    # 현재 initialize_agent는 deprecated
    grand_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    # [ ] question for python_agent
    question_message = """
    generate and save in current working directory 15 QRcodes that point to www.udemy.com/course/langchain.
    you have qrcode package installed already.
    """

    # [v] question for csv_agent
    question_message = """
    print seasons ascending order of the number of episodes they have.
    TIPS: Run the code script at once if possible.
    """
    result = grand_agent.invoke(question_message)
    print(f"{result=}")


if __name__ == "__main__":
    main()
