import os
from dotenv import load_dotenv

import pyupbit
import pandas as pd

import json
from openai import OpenAI
import ta
from ta.utils import dropna

import time
import requests
from datetime import datetime, timedelta
import sqlite3
import logging
import re
import schedule

import numpy as np

import base64
from io import BytesIO

# .env 파일레 저장된 환경변수 불러오기(api key)
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API 키 설정
# openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Upbit client
upbit = pyupbit.Upbit(os.getenv("UPBIT_ACCESS_KEY"), os.getenv("UPBIT_SECRET_KEY"))

def initialize_db(db_path='trading_coin.db'):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    decision TEXT,
                    percentage INTEGER,
                    reason TEXT,
                    btc_balance REAL,
                    krw_balance REAL,
                    btc_avg_buy_price REAL,
                    btc_krw_price REAL
                );
            ''')
            conn.commit()

        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"Error initializing database: {e}")

def save_decision_to_db(decision, current_status, db_path='trading_coin.db'):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
        
            status_dict = json.loads(current_status)
            current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
            
            data_to_insert = (
                decision.get('decision'),
                decision.get('percentage', 100),
                decision.get('reason', ''),
                status_dict.get('btc_balance'),
                status_dict.get('krw_balance'),
                status_dict.get('btc_avg_buy_price'),
                current_price
            )
            
            cursor.execute('''
                INSERT INTO decisions (timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price, btc_krw_price)
                VALUES (datetime('now', 'localtime'), ?, ?, ?, ?, ?, ?, ?)
            ''', data_to_insert)
        
            conn.commit()
         
        logging.info("Decision saved to database successfully.")
    except Exception as e:
        logging.error(f"Error saving decision to database: {e}")

def fetch_last_decisions(db_path='trading_coin.db', num_decisions=10):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, decision, percentage, reason, btc_balance, krw_balance, btc_avg_buy_price FROM decisions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (num_decisions,))
            decisions = cursor.fetchall()

            if decisions:
                formatted_decisions = []
                for decision in decisions:
                    ts = datetime.strptime(decision[0], "%Y-%m-%d %H:%M:%S")
                    ts_millis = int(ts.timestamp() * 1000)
                    
                    formatted_decision = {
                        "timestamp": ts_millis,
                        "decision": decision[1],
                        "percentage": decision[2],
                        "reason": decision[3],
                        "btc_balance": decision[4],
                        "krw_balance": decision[5],
                        "btc_avg_buy_price": decision[6]
                    }
                    formatted_decisions.append(str(formatted_decision))
                return "\n".join(formatted_decisions)
            else:
                return "No decisions found."
    except Exception as e:
        logging.error(f"Error fetching last decisions: {e}")
        return "Error fetching last decisions."

def get_current_status():
    try:
        orderbook = pyupbit.get_orderbook(ticker="KRW-BTC")
        current_time = orderbook['timestamp']
        btc_balance = 0
        krw_balance = 0
        btc_avg_buy_price = 0
        balances = upbit.get_balances()
        for b in balances:
            if b['currency'] == "BTC":
                btc_balance = b['balance']
                btc_avg_buy_price = b['avg_buy_price']
            if b['currency'] == "KRW":
                krw_balance = b['balance']

        current_status = {
            'current_time': current_time, 
            'orderbook': orderbook, 
            'btc_balance': btc_balance, 
            'krw_balance': krw_balance, 
            'btc_avg_buy_price': btc_avg_buy_price
        }
        return json.dumps(current_status)
    except Exception as e:
        logging.error(f"Error getting current status: {e}")
        return json.dumps({})

def fetch_and_prepare_data():
    try:
        df_daily = pyupbit.get_ohlcv("KRW-BTC", "day", count=180) # 180, 30
        df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=168) # 168 < 7 days of hourly data, 24

        def add_indicators(df):

            # 볼린저 밴드 추가
            indicator_bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=20)

            df['bb_bbm'] = indicator_bb.bollinger_mavg()
            df['bb_bbh'] = indicator_bb.bollinger_hband()
            df['bb_bbl'] = indicator_bb.bollinger_lband()

            # RSI (relative strength index) 추가
            df['rsi']= ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

            # MACD (moving aversge convergence divergence) 추가
            macd = ta.trend.MACD(close=df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # 이동동평균선(단기, 장기)
            df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['ema_12'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()


            """
            # Stochastic Oscillator 추가
            stoch = ta.momentum.StochOscillator(
                high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['stock_k'] = stoch.stoch()
            df['stock_d'] = stoch.stoch_signal()
            """
            
            # Average true Range(ATR 추가)
            df['atr'] = ta.volatility.AverageTrueRange(
                high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            # On- Balance Volume (OBV 추가)
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(
                close=df['close'], volume=df['volume']).on_balance_volume()

           
            """
            df['SMA_10'] = ta.sma(df['close'], length=10)
            df['EMA_10'] = ta.ema(df['close'], length=10)
            df['RSI_14'] = ta.rsi(df['close'], length=14)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            df = df.join(stoch)
            ema_fast = df['close'].ewm(span=12, adjust=False).mean()
            ema_slow = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema_fast - ema_slow
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
            df['Middle_Band'] = df['close'].rolling(window=20).mean()
            std_dev = df['close'].rolling(window=20).std()
            df['Upper_Band'] = df['Middle_Band'] + (std_dev * 2)
            df['Lower_Band'] = df['Middle_Band'] - (std_dev * 2)
            """

            return df

        df_daily = dropna(df_daily)
        df_hourly = dropna(df_hourly)

        df_daily = add_indicators(df_daily)
        df_hourly = add_indicators(df_hourly)

        combined_df = pd.concat([df_daily, df_hourly], keys=['daily', 'hourly'])
        combined_data = combined_df.to_json(orient='split')

        return json.dumps(combined_data)
        
    except Exception as e:
        logging.error(f"Error fetching and preparing data: {e}")
        return json.dumps({})

def get_news_data():
    url = f"https://serpapi.com/search.json?engine=google_news&q=btc&api_key={os.getenv('SERPAPI_API_KEY')}"
    result = "No news data available."

    try:
        response = requests.get(url)
        response.raise_for_status()
        news_results = response.json().get('news_results', [])
        
        simplified_news = []
        for news_item in news_results:
            if 'stories' in news_item:
                for story in news_item['stories']:
                    simplified_news.append({
                        'title': story['title'],
                        'link': story['link'],
                        'source': story['source'],
                        'published': story['date']
                    })
            else:
                simplified_news.append({
                    'title': news_item['title'],
                    'link': news_item['link'],
                    'source': news_item['source'],
                    'published': news_item['date']
                })

        if simplified_news:
            result = simplified_news
        else:
            logging.info("No news results found.")
    except Exception as e:
        logging.error(f"Error fetching news data: {e}")

    return json.dumps(result)

def fetch_fear_and_greed_index():
    url = "https://api.alternative.me/fng/"
    result = "No Fear and Greed Index data available."
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        
        if data and 'data' in data and len(data['data']) > 0:
            current_value = data['data'][0]['value']
            current_classification = data['data'][0]['value_classification']
            result = f"{current_value} - {current_classification}"
            logging.info(f"Fear and Greed Index: {result}")
        else:
            logging.warning("No Fear and Greed Index data in response")
            
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Fear and Greed Index data: {e}")
    except Exception as e:
        logging.error(f"Unexpected error while fetching Fear and Greed Index: {e}")

    return result

def analyze_market_conditions():
    try:
        # 일봉/시간봉 데이터 가져오기
        df_daily = pyupbit.get_ohlcv("KRW-BTC", "day", count=180) #180, 30
        df_hourly = pyupbit.get_ohlcv("KRW-BTC", interval="minute60", count=168) #168,24
        
        # 이동평균선 계산
        df_daily['MA5'] = df_daily['close'].rolling(window=5).mean()
        df_daily['MA10'] = df_daily['close'].rolling(window=10).mean()
        df_daily['MA20'] = df_daily['close'].rolling(window=20).mean()
        df_daily['MA60'] = df_daily['close'].rolling(window=60).mean()
        
        # RSI 계산
        def get_rsi(df, periods=14):
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df_daily['RSI'] = get_rsi(df_daily)
        
        # 현재 가격과 지표들
        current_price = df_daily['close'].iloc[-1]
        ma5 = df_daily['MA5'].iloc[-1]
        ma10 = df_daily['MA10'].iloc[-1]
        ma20 = df_daily['MA20'].iloc[-1]
        ma60 = df_daily['MA60'].iloc[-1]
        rsi = df_daily['RSI'].iloc[-1]
        
        # 원이토티 방식의 시장 분석
        market_analysis = {
            'trend': {
                'short_term': 'bullish' if current_price > ma5 else 'bearish',
                'mid_term': 'bullish' if ma5 > ma20 else 'bearish',
                'long_term': 'bullish' if ma20 > ma60 else 'bearish'
            },
            'momentum': {
                'rsi': rsi,
                'rsi_signal': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'
            },
            'support_resistance': {
                'support_1': ma20,
                'support_2': ma60,
                'current_price': current_price
            }
        }
        
        return market_analysis
        
    except Exception as e:
        logging.error(f"Error analyzing market conditions: {e}")
        return None

def determine_investment_ratio(market_analysis, fear_greed_data):
    try:
        # 기본 투자 비율
        base_ratio = 0.3
        
        # 트렌드 점수 계산 (각 -1 ~ 1)
        trend_scores = {
            'short': 1 if market_analysis['trend']['short_term'] == 'bullish' else -1,
            'mid': 1 if market_analysis['trend']['mid_term'] == 'bullish' else -1,
            'long': 1 if market_analysis['trend']['long_term'] == 'bullish' else -1
        }
        
        # RSI 기반 조정
        rsi = market_analysis['momentum']['rsi']
        if rsi < 30:  # 과매도
            rsi_multiplier = 1.2
        elif rsi > 70:  # 과매수
            rsi_multiplier = 0.8
        else:
            rsi_multiplier = 1.0
        
        # 공포탐욕지수 반영
        try:
            fear_greed_value = int(fear_greed_data.split(' - ')[0])
            if fear_greed_value < 20:  # 극도의 공포
                fg_multiplier = 1.3
            elif fear_greed_value > 80:  # 극도의 탐욕
                fg_multiplier = 0.7
            else:
                fg_multiplier = 1.0
        except:
            fg_multiplier = 1.0
        
        # 최종 투자 비율 계산
        trend_avg = sum(trend_scores.values()) / len(trend_scores)
        ratio = base_ratio * (1 + trend_avg * 0.2) * rsi_multiplier * fg_multiplier
        
        # 투자 비율 제한 (10% ~ 50%)
        ratio = max(0.1, min(0.5, ratio))
        
        # 투자 결정 이유 생성
        reasons = []
        reasons.append(f"기본 투자 비율: {base_ratio*100:.1f}%")
        reasons.append(f"단기 트렌드: {market_analysis['trend']['short_term']}")
        reasons.append(f"중기 트렌드: {market_analysis['trend']['mid_term']}")
        reasons.append(f"장기 트렌드: {market_analysis['trend']['long_term']}")
        reasons.append(f"RSI: {rsi:.1f} ({market_analysis['momentum']['rsi_signal']})")
        reasons.append(f"공포탐욕지수: {fear_greed_data}")
        
        return {
            'ratio': ratio,
            'reasons': reasons
        }
        
    except Exception as e:
        logging.error(f"Error determining investment ratio: {e}")
        return {'ratio': 0.3, 'reasons': ['Error calculating ratio']}

def display_market_analysis(market_analysis, investment_decision):

    print("\n=== Market Analysis (원이토티 방식) ===")
    print("┌" + "─" * 70 + "┐")
    print(f"│ 트렌드 분석:")
    print(f"│   단기 트렌드: {market_analysis['trend']['short_term']:>20}")
    print(f"│   중기 트렌드: {market_analysis['trend']['mid_term']:>20}")
    print(f"│   장기 트렌드: {market_analysis['trend']['long_term']:>20}")
    print(f"│ 모멘텀:")
    print(f"│   RSI: {market_analysis['momentum']['rsi']:>20.2f}")
    print(f"│   RSI 신호: {market_analysis['momentum']['rsi_signal']:>20}")
    print(f"│ 가격 레벨:")
    print(f"│   현재가: {market_analysis['support_resistance']['current_price']:>20,.0f}")
    print(f"│   지지선1: {market_analysis['support_resistance']['support_1']:>20,.0f}")
    print(f"│   지지선2: {market_analysis['support_resistance']['support_2']:>20,.0f}")
    print("│" + "─" * 69 + "│")
    print(f"│ 투자 비율: {investment_decision['ratio']*100:>20.1f}%")
    print("│ 결정 이유:")
    for reason in investment_decision['reasons']:
        print(f"│   - {reason}")
    print("└" + "─" * 70 + "┘")

# analyze_data_with_gpt3 함수 수정

def analyze_data_with_gpt3(combined_data, news_data, fear_greed_data, last_decisions, current_status):
    try:
        # 원이토티 방식의 시장 분석 추가
        market_analysis = analyze_market_conditions()
        if market_analysis is None:
            raise ValueError("Market analysis failed")
            
        investment_decision = determine_investment_ratio(market_analysis, fear_greed_data)
        
        # 분석 결과 표시
        display_market_analysis(market_analysis, investment_decision)
        
        # GPT-3 분석을 위한 데이터 요약
        market_summary = {
            'trend': market_analysis['trend'],
            'rsi': market_analysis['momentum']['rsi'],
            'current_price': market_analysis['support_resistance']['current_price']
        }
        
        # 최근 뉴스만 선택 (최대 3개)
        try:
            news_summary = json.loads(news_data)[:3] if isinstance(news_data, str) else []
        except:
            news_summary = []
        
        # 최근 거래 결정만 선택 (최대 3개)
        decisions_summary = last_decisions.split('\n')[:3] if isinstance(last_decisions, str) else []
        
        # GPT-3 분석
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = (
            f"Based on the following summarized data, make a trading decision:\n\n"
            f"1. Market Analysis:\n{json.dumps(market_summary, indent=2)}\n\n"
            f"2. Investment Ratio: {investment_decision['ratio']*100:.1f}%\n"
            f"3. Fear and Greed Index: {fear_greed_data}\n"
            f"4. Recent News: {json.dumps(news_summary, indent=2)}\n"
            f"5. Recent Decisions: {json.dumps(decisions_summary, indent=2)}\n\n"
            "You must respond with a valid JSON object in exactly this format:\n"
            "{\n"
            '  "decision": "buy/sell/hold",\n'
            '  "percentage": number between 0-100,\n'
            '  "reason": "brief explanation"\n'
            "}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a cryptocurrency trading assistant. You must respond with a valid JSON object containing exactly these fields: decision (buy/sell/hold), percentage (0-100), and reason (brief explanation)."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content.strip()
        logging.info(f"GPT Response: {response_text}")
        
        try:
            decision = json.loads(response_text)
            required_fields = ['decision', 'percentage', 'reason']
            if not all(field in decision for field in required_fields):
                raise ValueError("Missing required fields in decision")
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}")
            decision = {
                "decision": "hold",
                "percentage": int(investment_decision['ratio'] * 100),
                "reason": "Error parsing GPT response"
            }
        
        # 원이토티 분석 기반 투자 비율 반영  ******************************************  persontage

        decision['percentage'] = int(investment_decision['ratio'] * 100)
        
        return decision
        
    except Exception as e:
        logging.error(f"Error in analysis: {e}")
        return {
            "decision": "hold",
            "percentage": 0,
            "reason": f"Error in analysis: {str(e)}"
        }

def execute_buy(jh_ratio):
    print("Attempting to buy BTC...")
    try:
        krw = upbit.get_balance("KRW")
        if krw > 5000:
            # result = upbit.buy_market_order("KRW-BTC", krw*0.9995*0.3)
            result = upbit.buy_market_order("KRW-BTC", krw*0.9995*jh_ratio)
            print("Buy order successful:", result)
    except Exception as e:
        print(f"Failed to execute buy order: {e}")

def execute_sell(jh_ratio):
    print("Attempting to sell BTC...")
    try:
        btc = upbit.get_balance("BTC")
        current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
        if current_price*btc > 5000:
            result = upbit.sell_market_order("KRW-BTC", btc*jh_ratio)
            # result = upbit.sell_market_order("KRW-BTC", btc*0.3)
            print("Sell order successful:", result)
    except Exception as e:
        print(f"Failed to execute sell order: {e}")

def display_performance_metrics(analysis):

    print("\n=== Trading Performance Metrics ===")
    print("┌" + "─" * 70 + "┐")
    print(f"│ Total Trades: {analysis['total_trades']:>55} │")
    print(f"│ Buy Trades: {analysis['buy_trades']:>56} │")
    print(f"│ Sell Trades: {analysis['sell_trades']:>55} │")
    print(f"│ Hold Trades: {analysis['hold_trades']:>55} │")
    print(f"│ Profitable Trades: {analysis['profitable_trades']:>50} │")
    print(f"│ Average Profit: {analysis['average_profit']:>51.2f}% │")
    print(f"│ Success Rate: {analysis['success_rate']:>53.2f}% │")
    print("└" + "─" * 70 + "┘")

def display_strategy_improvements(improvements):

    print("\n=== Strategy Improvement Suggestions ===")
    print("┌" + "─" * 88 + "┐")
    
    lines = improvements.split('\n')
    for line in lines:
        if line.strip():
            while len(line) > 86:
                split_point = line[:86].rfind(' ')
                if split_point == -1:
                    split_point = 86
                print(f"│ {line[:split_point]:<86} │")
                line = line[split_point:].strip()
            print(f"│ {line:<86} │")
    
    print("└" + "─" * 88 + "┘")
    print("")

    # 한글 번역 추가
    
    print("> Strategy Improvements (KR): ")
    # print(improvements)
    lines = translate_to_korean(improvements)
    print(lines)
   
    print("└" + "─" * 88 + "┘")


def display_current_trade_status(current_status):
    try:
        status = json.loads(current_status)
        btc_balance = float(status['btc_balance'])
        krw_balance = float(status['krw_balance'])
        avg_buy_price = float(status['btc_avg_buy_price'])
        
        # 현재 BTC 가격 조회
        current_price = pyupbit.get_orderbook(ticker="KRW-BTC")['orderbook_units'][0]["ask_price"]
        
        # 총 보유자산 계산 (KRW + BTC 평가금액)
        total_btc_value = btc_balance * current_price
        total_assets = krw_balance + total_btc_value
        
        # 수익률 계산
        if avg_buy_price > 0 and btc_balance > 0:
            profit_rate = ((current_price - avg_buy_price) / avg_buy_price) * 100
        else:
            profit_rate = 0.0

        print("\n=== Current Trading Status ===")
        print("┌" + "─" * 60 + "┐")
        print(f"│ BTC Balance: {btc_balance:>35.8f} BTC ")
        print(f"│ KRW Balance: {krw_balance:>35,.0f} KRW ")
        print(f"│ Avg Buy Price: {avg_buy_price:>33,.0f} KRW ")
        print(f"│ Current Price: {current_price:>33,.0f} KRW ")
        print(f"│ Total Assets: {total_assets:>34,.0f} KRW ")
        print(f"│ Profit Rate: {profit_rate:>35.2f} %  ")
        
        # 수익률에 따른 상태 표시
        status_line = "│ Status: "
        if profit_rate > 0:
            status_line += "🔺 Profit"
        elif profit_rate < 0:
            status_line += "🔻 Loss"
        else:
            status_line += "➖ Break-even"
        status_line = f"{status_line:} "
        print(status_line)
        
        print("└" + "─" * 60 + "┘")
    except Exception as e:
        logging.error(f"Error displaying trade status: {e}")

def generate_strategy_improvements(performance_analysis):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        prompt = f"""
        Based on the following trading performance analysis, suggest improvements to the trading strategy:

        Performance Metrics:
        - Total Trades: {performance_analysis['total_trades']}
        - Buy Trades: {performance_analysis['buy_trades']}
        - Sell Trades: {performance_analysis['sell_trades']}
        - Hold Trades: {performance_analysis['hold_trades']}
        - Profitable Trades: {performance_analysis['profitable_trades']}
        - Average Profit: {performance_analysis['average_profit']:.2f}%
        - Success Rate: {performance_analysis['success_rate']:.2f}%

        Please analyze these metrics and provide:
        1. Key observations about the current strategy
        2. Specific areas for improvement
        3. Concrete suggestions for strategy adjustments
        4. Risk management recommendations
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert cryptocurrency trading analyst. Provide detailed, actionable insights for improving trading strategy."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"Error generating strategy improvements: {e}")
        return None

def save_strategy_review(analysis, improvements):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_review_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== Trading Performance Analysis ===\n\n")
            for key, value in analysis.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write("\n=== Strategy Improvement Suggestions ===\n\n")
            f.write(improvements)
        
        logging.info(f"Strategy review saved to {filename}")
        return filename
    
    except Exception as e:
        logging.error(f"Error saving strategy review: {e}")
        return None

def translate_to_korean(text):
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a translator. Translate the given English text to Korean."
                },
                {"role": "user", "content": f"Translate this to Korean: {text}"}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return "번역 실패"

def display_trading_decision(decision):

    print("\n=== Trading Decision ===")
    print("┌" + "─" * 85 + "┐")
    print(f"│ Decision: {decision['decision']:>20} ")
    print(f"│ Percentage: {decision['percentage']:>17}% ")
    
    # 영문 reason 텍스트 줄바꿈 처리
    reason_en = decision['reason']
    print("│ Reason (EN): ")
    while len(reason_en) > 84:
        split_point = reason_en[:84].rfind(' ')
        if split_point == -1:
            split_point = 84
        print(f"│ {reason_en[:split_point]:<84} ")
        reason_en = reason_en[split_point:].strip()
    print(f"│ {reason_en:<84} ")
    print("└" + "─" * 85 + "┘")
    
    # 한글 번역 추가
   
    print("│ Reason (KR): ")
    reason_kr = translate_to_korean(decision['reason'])
    while len(reason_kr) > 64:
        split_point = reason_kr[:64].rfind(' ')
        if split_point == -1:
            split_point = 64
        print(f" {reason_kr[:split_point]:<64} ")
        reason_kr = reason_kr[split_point:].strip()
    print(f" {reason_kr:<64} ")
    
    print("")


def analyze_trading_performance():
    try:
        with sqlite3.connect('trading_coin.db') as conn:
            cursor = conn.cursor()
            
            # 최근 거래 기록 가져오기
            cursor.execute('''
                SELECT 
                    timestamp,
                    decision,
                    percentage,
                    reason,
                    btc_balance,
                    krw_balance,
                    btc_avg_buy_price,
                    btc_krw_price
                FROM decisions 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            trades = cursor.fetchall()

            if not trades:
                logging.info("No trading history available for analysis.")
                return {
                    'total_trades': 0,
                    'buy_trades': 0,
                    'sell_trades': 0,
                    'hold_trades': 0,
                    'profitable_trades': 0,
                    'average_profit': 0.0,
                    'success_rate': 0.0
                }

            # 성과 분석
            total_trades = len(trades)
            buy_trades = sum(1 for trade in trades if trade[1] == 'buy')
            sell_trades = sum(1 for trade in trades if trade[1] == 'sell')
            hold_trades = sum(1 for trade in trades if trade[1] == 'hold')

            # 수익성 분석
            profitable_trades = 0
            total_profit_percentage = 0
            last_buy_price = None
            
            for i in range(len(trades)-1):
                current_trade = trades[i]
                next_trade = trades[i+1]
                
                if current_trade[1] == 'sell' and last_buy_price:
                    profit_percentage = (current_trade[7] - last_buy_price) / last_buy_price * 100
                    total_profit_percentage += profit_percentage
                    if profit_percentage > 0:
                        profitable_trades += 1
                
                if current_trade[1] == 'buy':
                    last_buy_price = current_trade[7]

            # 분석 결과 생성
            analysis = {
                'total_trades': total_trades,
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'hold_trades': hold_trades,
                'profitable_trades': profitable_trades,
                'average_profit': total_profit_percentage / sell_trades if sell_trades > 0 else 0,
                'success_rate': (profitable_trades / sell_trades * 100) if sell_trades > 0 else 0
            }
            
            logging.info(f"Trading performance analysis completed: {analysis}")
            return analysis

    except Exception as e:
        logging.error(f"Error analyzing trading performance: {e}")
        return {
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'hold_trades': 0,
            'profitable_trades': 0,
            'average_profit': 0.0,
            'success_rate': 0.0
        }

"""

    while True:
        count += 1
            print(f"\n{'='*20} Trading Round {count} {'='*20}")
        if count > 1:        
            break
"""

def ai_trading():
        count = 0
        review_interval = 1  # 3회 거래마다 전략 검토
        
        # 전략 검토 및 개선
        if count % review_interval == 0:
            print("\n🔄 Performing Strategy Review...")
            performance_analysis = analyze_trading_performance()
            
            if performance_analysis:
                improvements = generate_strategy_improvements(performance_analysis)
                if improvements:
                    display_strategy_improvements(improvements)
                    review_file = save_strategy_review(performance_analysis, improvements)
                    logging.info(f"Strategy review saved to {review_file}")
        try:
            combined_data = fetch_and_prepare_data()
            news_data = get_news_data()
            fear_greed_data = fetch_fear_and_greed_index()
            last_decisions = fetch_last_decisions()
            current_status = get_current_status()

            # 현재 상태 표시
            display_current_trade_status(current_status)

            decision = analyze_data_with_gpt3(combined_data, news_data, fear_greed_data, last_decisions, current_status)
            save_decision_to_db(decision, current_status)

            # 새로운 함수로 거래 결정 표시
            display_trading_decision(decision)

            jh_ratio = decision['percentage']/100

            print(f">> Decision: {decision['decision']:>20} ")
            print(f">> jh_ratio: {jh_ratio:>20} ")

            try:
                if decision['decision'] == "buy":
                    execute_buy(jh_ratio)
                elif decision['decision'] == "sell":
                    execute_sell(jh_ratio)
            except Exception as e:
                print(f"Failed to execute trade: {e}")

        except Exception as e:
            logging.error(f"Error in main execution loop: {e}")

if __name__ == "__main__":

    initialize_db()

    # 중복 실행 방지를 위한 변수

    trading_in_progress = False

    def job():
        global trading_in_progress
        if trading_in_progress:
            logging.error(f"Trading is already in progress. skipping this run")
            return
        try:
            trading_in_progress = True
            ai_trading()
        except Exception as e:
            logging.error(f"Error in job: {e}")
        finally:
            trading_in_progress = False

    # 매 4시간마다 실행
    schedule.every(4).hours.do(job)
"""
    schedule.every().day.at("17:00").do(job)
    schedule.every().day.at("18:00").do(job)
    schedule.every().day.at("17:00").do(job)
    schedule.every().day.at("18:00").do(job)
"""
    # test
    #job()

while True:
        schedule.run_pending()
        time.sleep(1)