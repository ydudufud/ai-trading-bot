import os
import time
import json
import threading
import requests
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import ccxt
import pandas as pd
import numpy as np
import ta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ========== إعدادات التداول المتقدمة مع الذكاء الاصطناعي ==========
class TradingConfig:
    # المفاتيح - يمكن تغييرها من الواجهة
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')  # مفتاح الذكاء الاصطناعي
    
    # إعدادات التداول
    AUTO_EXECUTE = os.getenv('AUTO_EXECUTE', 'false').lower() == 'true'
    RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '1.0'))
    QUOTE_ASSET = os.getenv('QUOTE_ASSET', 'USDT')
    TRADING_PAIRS = os.getenv('TRADING_PAIRS', 'BTC/USDT,ETH/USDT,ADA/USDT,BNB/USDT').split(',')
    
    # إعدادات الذكاء الاصطناعي
    AI_ENABLED = os.getenv('AI_ENABLED', 'true').lower() == 'true'
    AI_MODEL = os.getenv('AI_MODEL', 'gpt-3.5-turbo')

class AdvancedAITradingBot:
    def __init__(self):
        self.config = TradingConfig()
        self.exchange = None
        self.is_running = False
        self.trading_thread = None
        self.user_commands = []
        self.trading_history = []
        self.setup_exchange()
        
    def setup_exchange(self):
        """تهيئة اتصال البورصة"""
        try:
            if self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
                self.exchange = ccxt.binance({
                    'apiKey': self.config.BINANCE_API_KEY,
                    'secret': self.config.BINANCE_API_SECRET,
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                })
                # اختبار الاتصال
                self.exchange.fetch_balance()
                self.log("✅ تم الاتصال بباينانس بنجاح")
                return True
            else:
                self.log("⚠️  مفاتيح باينانس غير موجودة - وضع المحاكاة")
                return False
        except Exception as e:
            self.log(f"❌ فشل الاتصال بباينانس: {str(e)}")
            return False

    def log(self, message, level="INFO"):
        """تسجيل الرسائل مع الطابع الزمني"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        # حفظ آخر 100 رسالة للعرض في الواجهة
        self.user_commands.insert(0, log_entry)
        if len(self.user_commands) > 100:
            self.user_commands.pop()

    # ========== الذكاء الاصطناعي المتقدم ==========
    def get_ai_analysis(self, symbol, technical_data):
        """الحصول على تحليل الذكاء الاصطناعي للسوق"""
        if not self.config.OPENAI_API_KEY or not self.config.AI_ENABLED:
            return "🤖 الذكاء الاصطناعي غير مفعل - أضف مفتاح OpenAI API في الإعدادات"
        
        try:
            prompt = f"""
            أنا مساعد تداول ذكي. قم بتحليل العملة {symbol} بناء على البيانات الفنية التالية:
            
            البيانات الفنية:
            - السعر الحالي: ${technical_data['price']:.2f}
            - RSI: {technical_data['rsi']} ({'مشترى زائد' if technical_data['rsi'] > 70 else 'مباع زائد' if technical_data['rsi'] < 30 else 'محايد'})
            - MACD: {technical_data['macd']:.4f}
            - موضع البولينجر: {technical_data['bb_position']}
            - نسبة الحجم: {technical_data['volume_ratio']:.2f}x
            - الاتجاه: {technical_data['trend']}
            - قوة الإشارة: {technical_data['signal_strength']}/4
            
            التحليل الحالي:
            - الإشارة: {technical_data['signal']}
            - الثقة: {technical_data['confidence']}
            
            قدم تحليلاً شاملاً باللغة العربية يتضمن:
            1. تقييم عام للسوق
            2. المخاطر المحتملة
            3. التوصية النهائية (شراء/بيع/انتظار)
            4- السبب وراء التوصية
            
            كن دقيقاً واحترافياً في التحليل.
            """
            
            headers = {
                "Authorization": f"Bearer {self.config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.config.AI_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "أنت خبير تداول محترف في الأسواق المالية. قدم تحليلات دقيقة وواقعية بناء على البيانات الفنية."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                ai_analysis = result['choices'][0]['message']['content'].strip()
                self.log(f"🤖 الذكاء الاصطناعي حلل {symbol}", "AI")
                return ai_analysis
            else:
                error_msg = f"❌ خطأ في الذكاء الاصطناعي: {response.status_code}"
                self.log(error_msg, "ERROR")
                return error_msg
                
        except Exception as e:
            error_msg = f"❌ خطأ في اتصال الذكاء الاصطناعي: {str(e)}"
            self.log(error_msg, "ERROR")
            return error_msg

    # ========== المؤشرات الفنية المتقدمة ==========
    def calculate_advanced_indicators(self, df):
        """حساب جميع المؤشرات الفنية المتقدمة"""
        try:
            # المتوسطات المتحركة
            df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # RSI
            df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Volume indicators
            df['volume_sma'] = ta.volume.volume_sma(df['volume'], window=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # إشارات التداول المخصصة
            df['buy_signal'] = (
                (df['rsi_14'] < 35) &
                (df['macd'] > df['macd_signal']) &
                (df['close'] < df['bb_lower']) &
                (df['volume_ratio'] > 1.2)
            )
            
            df['sell_signal'] = (
                (df['rsi_14'] > 65) |
                (df['macd'] < df['macd_signal']) |
                (df['close'] > df['bb_upper'])
            )
            
            # قوة الإشارة
            df['signal_strength'] = 0
            df.loc[df['buy_signal'], 'signal_strength'] += 1
            df.loc[df['rsi_14'] < 30, 'signal_strength'] += 1
            df.loc[df['macd'] > df['macd_signal'], 'signal_strength'] += 1
            df.loc[df['close'] < df['bb_lower'], 'signal_strength'] += 1
            
            return df
            
        except Exception as e:
            self.log(f"❌ خطأ في حساب المؤشرات: {str(e)}", "ERROR")
            return df

    def analyze_market(self, symbol='BTC/USDT'):
        """تحليل السوق المتقدم مع الذكاء الاصطناعي"""
        try:
            # جلب بيانات OHLCV
            ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # حساب المؤشرات
            df = self.calculate_advanced_indicators(df)
            
            # التحليل النهائي
            latest = df.iloc[-1]
            analysis = {
                'symbol': symbol,
                'price': latest['close'],
                'rsi': round(latest['rsi_14'], 2),
                'macd': round(latest['macd'], 4),
                'bb_position': 'وسط',
                'volume_ratio': round(latest['volume_ratio'], 2),
                'trend': 'صاعد' if latest['sma_20'] > latest['sma_50'] else 'هابط',
                'signal_strength': int(latest['signal_strength']),
                'signal': 'محايد',
                'confidence': 'منخفضة',
                'ai_analysis': ''
            }
            
            # تحديد موقع السعر في Bollinger Bands
            if latest['close'] < latest['bb_lower']:
                analysis['bb_position'] = 'أسفل'
            elif latest['close'] > latest['bb_upper']:
                analysis['bb_position'] = 'أعلى'
            
            # حساب الإشارة
            if analysis['signal_strength'] >= 3:
                analysis['signal'] = 'شراء'
                analysis['confidence'] = 'عالية'
            elif analysis['signal_strength'] >= 2:
                analysis['signal'] = 'شراء' 
                analysis['confidence'] = 'متوسطة'
            
            # الحصول على تحليل الذكاء الاصطناعي
            if self.config.AI_ENABLED and self.config.OPENAI_API_KEY:
                analysis['ai_analysis'] = self.get_ai_analysis(symbol, analysis)
            
            return analysis
            
        except Exception as e:
            self.log(f"❌ خطأ في تحليل {symbol}: {str(e)}", "ERROR")
            return None

    # ========== نظام الأوامر الصوتية/النصية ==========
    def process_command(self, command):
        """معالجة الأوامر من المستخدم"""
        try:
            command = command.lower().strip()
            self.log(f"🎯 أمر مستلم: {command}", "COMMAND")
            
            if 'ابحث عن' in command or 'تحليل' in command:
                symbols = {
                    'btc': 'BTC/USDT', 'eth': 'ETH/USDT', 'ada': 'ADA/USDT',
                    'bnb': 'BNB/USDT', 'xrp': 'XRP/USDT', 'dot': 'DOT/USDT',
                    'sol': 'SOL/USDT', 'matic': 'MATIC/USDT', 'link': 'LINK/USDT'
                }
                
                for name, symbol in symbols.items():
                    if name in command:
                        analysis = self.analyze_market(symbol)
                        if analysis:
                            response = self.format_analysis_response(analysis)
                        else:
                            response = f"❌ لا يمكن تحليل {symbol}"
                        break
                else:
                    response = "⚠️ الرجاء تحديد العملة (مثال: 'ابحث عن BTC' أو 'تحليل ETH')"
            
            elif 'شغل التداول' in command or 'ابدأ' in command:
                if self.start_trading():
                    response = "✅ تم بدء التداول التلقائي على السحابة"
                else:
                    response = "⚠️ التداول مشغّل مسبقاً"
            
            elif 'اوقف التداول' in command or 'توقف' in command:
                self.stop_trading()
                response = "⏹️ تم إيقاف التداول التلقائي"
            
            elif 'الرصيد' in command:
                balance = self.get_balance()
                if balance:
                    balance_str = "\n".join([f"{asset}: {amount:.8f}" for asset, amount in balance.items()])
                    response = f"💰 الرصيد:\n{balance_str}"
                else:
                    response = "❌ لا يمكن جلب الرصيد"
            
            elif 'الحالة' in command:
                status = "🟢 شغال" if self.is_running else "⏸️ متوقف"
                ai_status = "🟢 مفعل" if self.config.AI_ENABLED and self.config.OPENAI_API_KEY else "⭕ معطل"
                response = f"حالة التداول: {status}\nحالة الذكاء الاصطناعي: {ai_status}"
            
            elif 'تفعيل الذكاء' in command:
                self.config.AI_ENABLED = True
                response = "✅ تم تفعيل الذكاء الاصطناعي"
            
            elif 'تعطيل الذكاء' in command:
                self.config.AI_ENABLED = False
                response = "⭕ تم تعطيل الذكاء الاصطناعي"
            
            elif 'المساعدة' in command or 'help' in command:
                response = """
🤖 **الأوامر المتاحة:**
- `ابحث عن BTC` - تحليل البيتكوين بالذكاء الاصطناعي
- `تحليل ETH` - تحليل الإيثيريوم بالذكاء الاصطناعي  
- `شغل التداول` - بدء التداول التلقائي
- `اوقف التداول` - إيقاف التداول
- `الرصيد` - عرض الرصيد
- `الحالة` - عرض حالة التداول والذكاء الاصطناعي
- `تفعيل الذكاء` - تفعيل الذكاء الاصطناعي
- `تعطيل الذكاء` - تعطيل الذكاء الاصطناعي
- `المساعدة` - عرض هذه الرسالة
                """
            else:
                response = "🤖 لم أفهم الأمر. جرب: 'المساعدة' لرؤية الأوامر المتاحة"
            
            self.log(f"🤖 رد: {response}", "RESPONSE")
            return response
                
        except Exception as e:
            error_msg = f"❌ خطأ في معالجة الأمر: {str(e)}"
            self.log(error_msg, "ERROR")
            return error_msg

    def format_analysis_response(self, analysis):
        """تنسيق استجابة التحليل مع الذكاء الاصطناعي"""
        response = f"""
📊 **تحليل {analysis['symbol']}**

💰 **السعر:** ${analysis['price']:.2f}
📈 **RSI:** {analysis['rsi']}
📊 **MACD:** {analysis['macd']:.4f}
🎯 **الإشارة:** {analysis['signal']}
💪 **الثقة:** {analysis['confidence']}
📊 **الاتجاه:** {analysis['trend']}
💪 **قوة الإشارة:** {analysis['signal_strength']}/4

"""
        if analysis['ai_analysis'] and not analysis['ai_analysis'].startswith('❌'):
            response += f"🤖 **تحليل الذكاء الاصطناعي:**\n{analysis['ai_analysis']}"
        else:
            response += "🤖 *الذكاء الاصطناعي غير متوفر*"
        
        return response

    # ========== نظام التداول التلقائي ==========
    def trading_loop(self):
        """الحلقة الرئيسية للتداول التلقائي"""
        self.log("🚀 بدء التداول التلقائي على السحابة...", "SYSTEM")
        
        while self.is_running:
            try:
                for symbol in self.config.TRADING_PAIRS:
                    if not self.is_running:
                        break
                    
                    analysis = self.analyze_market(symbol)
                    if analysis and analysis['signal'] == 'شراء' and analysis['confidence'] == 'عالية':
                        self.execute_trade_signal(symbol, analysis)
                    
                    time.sleep(2)
                
                # انتظار 5 دقائق بين كل دورة
                self.log("🔍 جولة الفحص اكتملت، انتظار 5 دقائق...", "SYSTEM")
                for i in range(300):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.log(f"❌ خطأ في حلقة التداول: {str(e)}", "ERROR")
                time.sleep(60)

    def execute_trade_signal(self, symbol, analysis):
        """تنفيذ إشارة التداول"""
        try:
            if self.config.AUTO_EXECUTE and self.exchange:
                # حساب حجم الصفقة
                balance = self.exchange.fetch_balance()
                usdt_balance = balance['total'].get('USDT', 0)
                
                if usdt_balance > 10:
                    risk_amount = usdt_balance * (self.config.RISK_PERCENTAGE / 100)
                    price = analysis['price']
                    quantity = risk_amount / price
                    
                    # تنفيذ الشراء
                    order = self.exchange.create_market_buy_order(symbol, quantity)
                    
                    # تسجيل الصفقة
                    trade_info = {
                        'symbol': symbol,
                        'side': 'buy',
                        'quantity': quantity,
                        'price': price,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.trading_history.append(trade_info)
                    
                    self.log(f"✅ تم شراء {quantity:.6f} {symbol} بسعر ${price:.2f}", "TRADE")
            
            else:
                self.log(f"💡 [محاكاة] إشارة شراء لـ {symbol} - السعر: ${analysis['price']:.2f}", "SIMULATION")
                
        except Exception as e:
            self.log(f"❌ خطأ في تنفيذ الصفقة: {str(e)}", "ERROR")

    def start_trading(self):
        """بدء التداول التلقائي"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()
            return True
        return False

    def stop_trading(self):
        """إيقاف التداول التلقائي"""
        self.is_running = False
        self.log("⏹️ تم إيقاف التداول التلقائي", "SYSTEM")

    def get_balance(self):
        """جلب الرصيد"""
        try:
            if self.exchange:
                balance = self.exchange.fetch_balance()
                return {asset: amount for asset, amount in balance['total'].items() if amount > 0.00000001}
            return {}
        except Exception as e:
            self.log(f"❌ خطأ في جلب الرصيد: {str(e)}", "ERROR")
            return {}

    def update_config(self, new_config):
        """تحديث الإعدادات من واجهة الويب"""
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    # تحويل القيم إلى الأنواع المناسبة
                    if key in ['AUTO_EXECUTE', 'AI_ENABLED']:
                        value = value.lower() == 'true'
                    elif key in ['RISK_PERCENTAGE']:
                        value = float(value)
                    elif key in ['TRADING_PAIRS']:
                        value = value.split(',')
                    
                    setattr(self.config, key, value)
            
            # إعادة تهيئة الاتصال إذا تم تغيير المفاتيح
            if 'BINANCE_API_KEY' in new_config or 'BINANCE_API_SECRET' in new_config:
                self.setup_exchange()
            
            self.log("✅ تم تحديث الإعدادات بنجاح", "SYSTEM")
            return True
        except Exception as e:
            self.log(f"❌ خطأ في تحديث الإعدادات: {str(e)}", "ERROR")
            return False

# إنشاء الكائن العالمي
trading_bot = AdvancedAITradingBot()

# ========== واجهة الويب ==========
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>🤖 AI Trading Bot - السحابة</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #0f1419; 
            color: white; 
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
        }
        .card { 
            background: #1e2328; 
            padding: 20px; 
            margin: 10px 0; 
            border-radius: 10px; 
            border: 1px solid #333; 
        }
        .btn { 
            background: #00d2d2; 
            color: white; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 5px; 
        }
        .btn-danger { background: #ff4444; }
        .btn-success { background: #00c853; }
        .btn-ai { background: #9c27b0; }
        .form-group { margin: 10px 0; }
        input, select { 
            width: 100%; 
            padding: 8px; 
            margin: 5px 0; 
            background: #2a2e35; 
            border: 1px solid #444; 
            color: white; 
            border-radius: 4px; 
        }
        .logs { 
            background: black; 
            color: #00ff00; 
            padding: 15px; 
            border-radius: 5px; 
            height: 200px; 
            overflow-y: scroll; 
            font-family: monospace; 
        }
        .status-running { color: #00ff00; }
        .status-stopped { color: #ff4444; }
        .command-result { 
            background: #2a2e35; 
            padding: 15px; 
            border-radius: 8px; 
            margin-top: 10px; 
            white-space: pre-line; 
        }
        .ai-analysis {
            background: #2d1b69;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
            border-left: 4px solid #9c27b0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Trading Bot - السحابة</h1>
        
        <!-- بطاقة التحكم السريع -->
        <div class="card">
            <h3>🎮 التحكم السريع</h3>
            <button class="btn btn-success" onclick="startTrading()">▶️ بدء التداول</button>
            <button class="btn btn-danger" onclick="stopTrading()">⏹️ إيقاف التداول</button>
            <button class="btn" onclick="getBalance()">💰 الرصيد</button>
            <button class="btn btn-ai" onclick="toggleAI()">🤖 تفعيل/تعطيل الذكاء</button>
            <span id="status" class="status-stopped">⏸️ متوقف</span>
            <span id="aiStatus" style="margin-left: 20px;">🤖 الذكاء: ⭕ معطل</span>
        </div>

        <!-- بطاقة الأوامر الصوتية -->
        <div class="card">
            <h3>🎤 الأوامر الذكية</h3>
            <input type="text" id="commandInput" placeholder="اكتب أمر مثل: 'ابحث عن BTC' أو 'شغل التداول'">
            <button class="btn" onclick="sendCommand()">🚀 تنفيذ الأمر</button>
            <div id="commandResult" class="command-result"></div>
        </div>

        <!-- بطاقة التحليل السريع -->
        <div class="card">
            <h3>📊 التحليل الفني السريع</h3>
            <button class="btn" onclick="analyzeMarket('BTC/USDT')">تحليل BTC</button>
            <button class="btn" onclick="analyzeMarket('ETH/USDT')">تحليل ETH</button>
            <button class="btn" onclick="analyzeMarket('ADA/USDT')">تحليل ADA</button>
            <button class="btn" onclick="analyzeMarket('BNB/USDT')">تحليل BNB</button>
            <div id="analysisResult" style="margin-top: 10px;"></div>
        </div>

        <!-- بطاقة الإعدادات -->
        <div class="card">
            <h3>⚙️ الإعدادات المتقدمة</h3>
            <form id="configForm">
                <div class="form-group">
                    <label>مفتاح API الباينانس:</label>
                    <input type="text" name="BINANCE_API_KEY" value="{{ config.BINANCE_API_KEY }}" placeholder="أدخل المفتاح هنا">
                </div>
                <div class="form-group">
                    <label>الرمز السري للباينانس:</label>
                    <input type="password" name="BINANCE_API_SECRET" value="{{ config.BINANCE_API_SECRET }}" placeholder="أدخل الرمز السري">
                </div>
                <div class="form-group">
                    <label>مفتاح OpenAI API:</label>
                    <input type="password" name="OPENAI_API_KEY" value="{{ config.OPENAI_API_KEY }}" placeholder="أدخل مفتاح الذكاء الاصطناعي">
                </div>
                <div class="form-group">
                    <label>تفعيل الذكاء الاصطناعي:</label>
                    <select name="AI_ENABLED">
                        <option value="true" {% if config.AI_ENABLED %}selected{% endif %}>مفعل</option>
                        <option value="false" {% if not config.AI_ENABLED %}selected{% endif %}>معطل</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>نسبة المخاطرة %:</label>
                    <input type="number" name="RISK_PERCENTAGE" value="{{ config.RISK_PERCENTAGE }}" step="0.1" min="0.1" max="10">
                </div>
                <div class="form-group">
                    <label>التنفيذ التلقائي:</label>
                    <select name="AUTO_EXECUTE">
                        <option value="false" {% if not config.AUTO_EXECUTE %}selected{% endif %}>محاكاة</option>
                        <option value="true" {% if config.AUTO_EXECUTE %}selected{% endif %}>حقيقي</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>العملات للمراقبة:</label>
                    <input type="text" name="TRADING_PAIRS" value="{{ config.TRADING_PAIRS | join(',') }}" placeholder="BTC/USDT,ETH/USDT,ADA/USDT">
                </div>
                <button type="submit" class="btn">💾 حفظ الإعدادات</button>
            </form>
        </div>

        <!-- بطاقة السجلات -->
        <div class="card">
            <h3>📝 سجلات النظام</h3>
            <div class="logs" id="logs">
                {% for log in logs %}
                <div>{{ log }}</div>
                {% endfor %}
            </div>
            <button class="btn" onclick="clearLogs()">🗑️ مسح السجلات</button>
        </div>
    </div>

    <script>
        // تحديث الحالة
        function updateStatus() {
            fetch('/status').then(r => r.json()).then(data => {
                const statusEl = document.getElementById('status');
                statusEl.className = data.running ? 'status-running' : 'status-stopped';
                statusEl.textContent = data.running ? '🟢 شغال' : '⏸️ متوقف';
                
                const aiStatusEl = document.getElementById('aiStatus');
                aiStatusEl.textContent = data.ai_enabled ? '🤖 الذكاء: 🟢 مفعل' : '🤖 الذكاء: ⭕ معطل';
                aiStatusEl.style.color = data.ai_enabled ? '#00ff00' : '#ff4444';
            });
        }

        // التحكم في التداول
        function startTrading() {
            fetch('/start', {method: 'POST'}).then(r => r.json()).then(data => {
                alert(data.message);
                updateStatus();
            });
        }

        function stopTrading() {
            fetch('/stop', {method: 'POST'}).then(r => r.json()).then(data => {
                alert(data.message);
                updateStatus();
            });
        }

        function toggleAI() {
            fetch('/toggle_ai', {method: 'POST'}).then(r => r.json()).then(data => {
                alert(data.message);
                updateStatus();
            });
        }

        function getBalance() {
            fetch('/balance').then(r => r.json()).then(data => {
                if (Object.keys(data).length > 0) {
                    let balanceText = '💰 الرصيد:\\n';
                    for (const [asset, amount] of Object.entries(data)) {
                        balanceText += `${asset}: ${parseFloat(amount).toFixed(8)}\\n`;
                    }
                    alert(balanceText);
                } else {
                    alert('❌ لا يوجد رصيد أو خطأ في الاتصال');
                }
            });
        }

        // الأوامر الذكية
        function sendCommand() {
            const command = document.getElementById('commandInput').value;
            if (!command.trim()) return;
            
            fetch('/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: command})
            }).then(r => r.json()).then(data => {
                document.getElementById('commandResult').innerHTML = data.response.replace(/\\n/g, '<br>');
                document.getElementById('commandInput').value = '';
            });
        }

        // تحليل السوق
        function analyzeMarket(symbol) {
            fetch('/analyze/' + encodeURIComponent(symbol))
                .then(r => r.json())
                .then(data => {
                    const resultEl = document.getElementById('analysisResult');
                    if (data.error) {
                        resultEl.innerHTML = `<div style="color: #ff4444">${data.error}</div>`;
                    } else {
                        resultEl.innerHTML = formatAnalysis(data);
                    }
                });
        }

        function formatAnalysis(analysis) {
            let html = `
                <div style="background: #2a2e35; padding: 15px; border-radius: 8px; margin-bottom: 10px;">
                    <strong>${analysis.symbol}</strong><br>
                    السعر: $${analysis.price.toFixed(2)}<br>
                    RSI: ${analysis.rsi}<br>
                    الإشارة: ${analysis.signal}<br>
                    الثقة: ${analysis.confidence}<br>
                    قوة الإشارة: ${analysis.signal_strength}/4
                </div>
            `;
            
            if (analysis.ai_analysis && !analysis.ai_analysis.includes('❌')) {
                html += `<div class="ai-analysis">🤖 <strong>تحليل الذكاء الاصطناعي:</strong><br>${analysis.ai_analysis.replace(/\\n/g, '<br>')}</div>`;
            }
            
            return html;
        }

        // إرسال الإعدادات
        document.getElementById('configForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            }).then(r => r.json()).then(data => {
                alert(data.message);
                updateStatus();
            });
        });

        // تحديث السجلات تلقائياً
        function updateLogs() {
            fetch('/logs').then(r => r.json()).then(data => {
                const logsContainer = document.getElementById('logs');
                logsContainer.innerHTML = data.logs.map(log => 
                    `<div>${log}</div>`
                ).join('');
                logsContainer.scrollTop = logsContainer.scrollHeight;
            });
        }

        function clearLogs() {
            fetch('/clear_logs', {method: 'POST'}).then(() => updateLogs());
        }

        // السماح بالضغط على Enter في حقل الأوامر
        document.getElementById('commandInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendCommand();
            }
        });

        // تحديث كل 5 ثواني
        setInterval(updateStatus, 5000);
        setInterval(updateLogs, 5000);
        updateStatus();
        updateLogs();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, 
        config=trading_bot.config,
        logs=trading_bot.user_commands[:20]
    )

@app.route('/status')
def status():
    return jsonify({
        "running": trading_bot.is_running,
        "ai_enabled": trading_bot.config.AI_ENABLED and bool(trading_bot.config.OPENAI_API_KEY)
    })

@app.route('/start', methods=['POST'])
def start_trading():
    if trading_bot.start_trading():
        return jsonify({"message": "تم بدء التداول التلقائي"})
    return jsonify({"message": "التداول مشغّل مسبقاً"})

@app.route('/stop', methods=['POST'])
def stop_trading():
    trading_bot.stop_trading()
    return jsonify({"message": "تم إيقاف التداول التلقائي"})

@app.route('/toggle_ai', methods=['POST'])
def toggle_ai():
    trading_bot.config.AI_ENABLED = not trading_bot.config.AI_ENABLED
    status = "مفعل" if trading_bot.config.AI_ENABLED else "معطل"
    return jsonify({"message": f"تم {status} الذكاء الاصطناعي"})

@app.route('/balance')
def balance():
    return jsonify(trading_bot.get_balance())

@app.route('/command', methods=['POST'])
def process_command():
    data = request.get_json()
    command = data.get('command', '')
    response = trading_bot.process_command(command)
    return jsonify({"response": response})

@app.route('/analyze/<path:symbol>')
def analyze_symbol(symbol):
    analysis = trading_bot.analyze_market(symbol)
    return jsonify(analysis if analysis else {"error": "فشل التحليل"})

@app.route('/config', methods=['POST'])
def update_config():
    data = request.get_json()
    if trading_bot.update_config(data):
        return jsonify({"message": "تم تحديث الإعدادات بنجاح"})
    return jsonify({"message": "فشل تحديث الإعدادات"})

@app.route('/logs')
def get_logs():
    return jsonify({"logs": trading_bot.user_commands[:50]})

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    trading_bot.user_commands.clear()
    return jsonify({"message": "تم مسح السجلات"})

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "4.0",
        "features": [
            "التداول الآلي 24/7",
            "الذكاء الاصطناعي المتقدم", 
            "المؤشرات الفنية المتقدمة",
            "نظام الأوامر الذكية",
            "واجهة ويب متكاملة"
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
