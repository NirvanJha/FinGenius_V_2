from dotenv import load_dotenv
from pydantic import BaseModel

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from langchain.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from openai import OpenAI

import os

import yfinance as yf

load_dotenv()

app = FastAPI()

# LangChain ChatOpenAI instance (currently unused for requests; kept for future extension)
model = ChatOpenAI(
    api_key=os.getenv("THESYS_API_KEY"),
    model='c1/openai/gpt-5/v-20250930',
    base_url='https://api.thesys.dev/v1/embed/',
)

# Direct OpenAI client against Thesys for robust compatibility
thesys_client = OpenAI(
    api_key=os.getenv("THESYS_API_KEY"),
    base_url='https://api.thesys.dev/v1/embed/',
)

checkpointer = InMemorySaver()


@tool('get_stock_price', description='A function that returns the current stock price based on a ticker symbol.')
def get_stock_price(ticker: str):
    print('get_stock_price tool is being used')
    stock = yf.Ticker(ticker)
    return stock.history()['Close'].iloc[-1]


@tool('get_historical_stock_price', description='A function that returns the current stock price over time based on a ticker symbol and a start and end date.')
def get_historical_stock_price(ticker: str, start_date: str, end_date: str):
    print('get_historical_stock_price tool is being used')
    stock = yf.Ticker(ticker)
    return stock.history(start=start_date, end=end_date).to_dict()


@tool('get_balance_sheet', description='A function that returns the balance sheet based on a ticker symbol.')
def get_balance_sheet(ticker: str):
    print('get_balance_sheet tool is being used')
    stock = yf.Ticker(ticker)
    return stock.balance_sheet


@tool('get_stock_news', description='A function that returns news based on a ticker symbol.')
def get_stock_news(ticker: str):
    print('get_stock_news tool is being used')
    stock = yf.Ticker(ticker)
    return stock.news
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a stock analysis assistant. You have the ability to get real-time "
            "stock prices, historical stock prices (given a date range), news and balance "
            "sheet data for a given ticker symbol. "
            "You can use tools to look up information.\n\n"
            "Available tools: {tool_names}\n\n"
            "{tools}\n\n"
            "When you decide to use a tool, think step-by-step.",
        ),
        # Optional chat history placeholder (can be left empty when calling)
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        # The agent_scratchpad is where previous tool invocations & thoughts are injected
        ("assistant", "{agent_scratchpad}"),
    ]
)

agent = create_react_agent(
    model,
    [get_stock_price, get_historical_stock_price, get_balance_sheet, get_stock_news],
    prompt,
)


class PromptObject(BaseModel):
    content: str
    id: str
    role: str


class RequestObject(BaseModel):
    prompt: PromptObject
    threadId: str
    responseId: str


@app.post('/api/chat')
async def chat(request: RequestObject):
    def generate():
        # Call Thesys directly via the OpenAI-compatible client to avoid LangChain
        # stop/stream compatibility issues.
        response = thesys_client.chat.completions.create(
            model='c1/openai/gpt-5/v-20250930',
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a stock analysis assistant. You have the ability to get "
                        "real-time stock prices, historical stock prices (given a date range), "
                        "news and balance sheet data for a given ticker symbol."
                    ),
                },
                {"role": "user", "content": request.prompt.content},
            ],
        )
        # Send back the first choice content as a single streamed chunk
        content = response.choices[0].message.content
        yield content

    return StreamingResponse(
        generate(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache, no-transform',
            'Connection': 'keep-alive',
        },
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)