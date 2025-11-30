import os
import time
import json
import threading
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
import ccxt
import pandas as pd
import numpy as np
import ta
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# إعدادات التداول القابلة للتخصيص
class TradingConfig:
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    AI_API_KEY = os.getenv('AI_API_KEY', '')
    AUTO_EXECUTE = os.getenv('AUTO_EXECUTE', 'false').lower() == 'true'
    RISK_PCT = float(os.getenv('RISK_PCT', '0.01'))
    QUOTE_ASSET = 'USDT'

class AITradingBot:
    def __init__(self):
        self.config = TradingConfig()
        self.exchange = None
        self.is_running = False
        self.trading_thread = None
        self.user_commands = []
        self.setup_exchange()
        
    def setup_exchange(self):
        """تهيئة اتصال باينانس"""
        try:
            if self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
                self.exchange = ccxt.binance({
                    'apiKey': self.config.BINANCE_API_KEY,
                    'secret': self.config.BINANCE_API_SECRET,
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })
                self.exchange.fetch_balance()
                self.log("✅ تم الاتصال بباينانس بنجاح")
                return True
            else:
                self.log("❌ مفاتيح باينانس غير موجودة - وضع المحاكاة")
                return False
        except Exception as e:
            self.log(f"❌ فشل الاتصال بباينانس: {e}")
            return False
    
    def log(self, message):
        """تسجيل الرسائل"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        self.user_commands.insert(0, log_message)
        if len(self.user_commands) > 100:
            self.user_commands.pop()
    
    def fetch_symbols(self):
        """جلب قائمة العملات"""
        try:
            markets = self.exchange.load_markets()
            symbols = [s for s in markets if s.endswith(f"/{self.config.QUOTE_ASSET}")]
            return symbols[:20]
        except Exception as e:
            self.log(f"❌ خطأ في جلب الرموز: {e}")
            return []
    
    def fetch_ohlcv(self, symbol, timeframe="1h", limit=100):
        """جلب بيانات التداول"""
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(data, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            self.log(f"❌ خطأ في جلب بيانات {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df):
        """حساب المؤشرات الفنية"""
        try:
            # المتوسطات المتحركة
            df['ema20'] = ta.trend.ema_indicator(df['close'], window=20)
            df['ema50'] = ta.trend.ema_indicator(df['close'], window=50)
            
            # RSI
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
            
            # Bollinger Bands
            df['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            
            # Volume SMA
            df['volume_sma'] = df['volume'].rolling(20).mean()
            
            return df
        except Exception as e:
            self.log(f"❌ خطأ في حساب المؤشرات: {e}")
            return df
    
    def analyze_symbol(self, symbol):
        """تحليل عملة واحدة"""
        try:
            df = self.fetch_ohlcv(symbol, "1h", 100)
            if df.empty:
                return None
            
            df = self.calculate_indicators(df)
            last = df.iloc[-1]
            
            analysis = {
                'symbol': symbol,
                'price': float(last['close']),
                'volume': float(last['volume']),
                'ema20': float(last['ema20']) if not np.isnan(last['ema20']) else 0,
                'ema50': float(last['ema50']) if not np.isnan(last['ema50']) else 0,
                'rsi': float(last['rsi']) if not np.isnan(last['rsi']) else 50,
                'macd': float(last['macd']) if not np.isnan(last['macd']) else 0,
                'signal': 'hold',
                'confidence': 'medium'
            }
            
            # إشارات تداول متقدمة
            buy_signals = 0
            if analysis['ema20'] > analysis['ema50']:
                buy_signals += 1
            if 30 < analysis['rsi'] < 70:
                buy_signals += 1
            if analysis['macd'] > 0:
                buy_signals += 1
            
            if buy_signals >= 2:
                analysis['signal'] = 'buy'
                analysis['confidence'] = 'high' if buy_signals == 3 else 'medium'
            
            return analysis
            
        except Exception as e:
            self.log(f"❌ خطأ في تحليل {symbol}: {e}")
            return None

    def process_command(self, command):
        """معالجة الأوامر من المستخدم"""
        try:
            command = command.lower().strip()
            self.log(f"🎯 أمر مستلم: {command}")
            
            if 'ابحث عن' in command or 'تحليل' in command:
                # استخراج اسم العملة من الأمر
                symbols = ['btc', 'eth', 'ada', 'dot', 'link', 'bnb', 'xrp']
                symbol_found = None
                for s in symbols:
                    if s in command:
                        symbol_found = f"{s.upper()}/USDT"
                        break
                
                if symbol_found:
                    analysis = self.analyze_symbol(symbol_found)
                    if analysis:
                        response = (f"📊 تحليل {symbol_found}:\n"
                                  f"💰 السعر: ${analysis['price']:.2f}\n"
                                  f"📈 RSI: {analysis['rsi']:.1f}\n"
                                  f"🎯 الإشارة: {analysis['signal']}\n"
                                  f"💪 الثقة: {analysis['confidence']}")
                    else:
                        response = f"❌ لا يمكن تحليل {symbol_found}"
                else:
                    response = "⚠️ الرجاء تحديد العملة (مثال: 'ابحث عن BTC')"
            
            elif 'شغل التداول' in command or 'ابدأ' in command:
                if self.start_trading():
                    response = "✅ تم بدء التداول التلقائي"
                else:
                    response = "⚠️ التداول مشغّل مسبقاً"
            
            elif 'اوقف التداول' in command or 'توقف' in command:
                self.stop_trading()
                response = "⏹️ تم إيقاف التداول التلقائي"
            
            elif 'الرصيد' in command:
                balance = self.get_balance()
                response = f"💰 الرصيد: {json.dumps(balance, ensure_ascii=False)}"
            
            elif 'السجلات' in command:
                response = "📝 استخدم واجهة الويب لمشاهدة السجلات الكاملة"
            
            else:
                response = "🤖 لم أفهم الأمر. جرب: 'ابحث عن BTC' أو 'شغل التداول' أو 'الرصيد'"
            
            self.log(f"🤖 رد: {response}")
            return response
                
        except Exception as e:
            error_msg = f"❌ خطأ في معالجة الأمر: {str(e)}"
            self.log(error_msg)
            return error_msg
    
    def execute_trade(self, symbol, action, quantity):
        """تنفيذ صفقة"""
        if not self.config.AUTO_EXECUTE:
            self.log(f"💡 [محاكاة] {action.upper()} {symbol} الكمية: {quantity:.6f}")
            return {"status": "dry_run"}
        
        try:
            order = self.exchange.create_order(symbol, 'market', action, quantity)
            self.log(f"✅ تم تنفيذ أمر {action} لـ {symbol}")
            return {"status": "success", "order": order}
        except Exception as e:
            self.log(f"❌ خطأ في تنفيذ الأمر: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_balance(self):
        """جلب الرصيد"""
        try:
            if self.exchange:
                balance = self.exchange.fetch_balance()
                return {k: v for k, v in balance['total'].items() if v > 0}
            return {}
        except Exception as e:
            self.log(f"❌ خطأ في جلب الرصيد: {e}")
            return {}
    
    def trading_loop(self):
        """الحلقة الرئيسية للتداول"""
        self.log("🚀 بدء التداول التلقائي على السحابة...")
        
        while self.is_running:
            try:
                symbols = self.fetch_symbols()
                self.log(f"🔍 فحص {len(symbols)} عملة...")
                
                for symbol in symbols:
                    if not self.is_running:
                        break
                    
                    analysis = self.analyze_symbol(symbol)
                    if analysis and analysis['signal'] == 'buy' and analysis['confidence'] == 'high':
                        # حساب الكمية (نسبة المخاطرة)
                        balance = self.get_balance()
                        usdt_balance = balance.get('USDT', 0)
                        
                        if usdt_balance > 10:
                            quantity = (usdt_balance * self.config.RISK_PCT) / analysis['price']
                            self.execute_trade(symbol, "buy", quantity)
                    
                    time.sleep(1)
                
                # انتظار 5 دقائق بين كل مسح
                for i in range(300):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.log(f"❌ خطأ في حلقة التداول: {e}")
                time.sleep(60)
    
    def start_trading(self):
        """بدء التداول"""
        if not self.is_running:
            self.is_running = True
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()
            self.log("🎯 تم بدء التداول التلقائي")
            return True
        return False
    
    def stop_trading(self):
        """إيقاف التداول"""
        self.is_running = False
        self.log("⏹️ تم إيقاف التداول التلقائي")

    def update_config(self, new_config):
        """تحديث الإعدادات من واجهة الويب"""
        try:
            for key, value in new_config.items():
                if hasattr(self.config, key):
                    # تحويل القيم إلى الأنواع المناسبة
                    if key in ['AUTO_EXECUTE']:
                        value = value.lower() == 'true'
                    elif key in ['RISK_PCT']:
                        value = float(value)
                    setattr(self.config, key, value)
            
            # إعادة تهيئة الاتصال إذا تم تغيير المفاتيح
            if 'BINANCE_API_KEY' in new_config or 'BINANCE_API_SECRET' in new_config:
                self.setup_exchange()
            
            self.log("✅ تم تحديث الإعدادات بنجاح")
            return True
        except Exception as e:
            self.log(f"❌ خطأ في تحديث الإعدادات: {e}")
            return False

# إنشاء الكائن العالمي
trading_bot = AITradingBot()

# واجهة الويب
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html dir="rtl">
<head>
    <meta charset="UTF-8">
    <title>🤖 AI Trading Bot - السحابة</title>
    <style>
        body { font-family: Arial; margin: 0; padding: 20px; background: #0f1419; color: white; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: #1e2328; padding: 20px; margin: 10px 0; border-radius: 10px; border: 1px solid #333; }
        .btn { background: #00d2d2; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn-danger { background: #ff4444; }
        .btn-success { background: #00c853; }
        .form-group { margin: 10px 0; }
        input, select { width: 100%; padding: 8px; margin: 5px 0; background: #2a2e35; border: 1px solid #444; color: white; border-radius: 4px; }
        .logs { background: black; color: #00ff00; padding: 15px; border-radius: 5px; height: 200px; overflow-y: scroll; font-family: monospace; }
        .status-running { color: #00ff00; }
        .status-stopped { color: #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI Trading Bot - التحكم الكامل</h1>
        
        <!-- بطاقة التحكم السريع -->
        <div class="card">
            <h3>🎮 التحكم السريع</h3>
            <button class="btn btn-success" onclick="startTrading()">▶️ بدء التداول</button>
            <button class="btn btn-danger" onclick="stopTrading()">⏹️ إيقاف التداول</button>
            <button class="btn" onclick="getBalance()">💰 الرصيد</button>
            <span id="status" class="status-stopped">⏸️ متوقف</span>
        </div>

        <!-- بطاقة الأوامر الصوتية -->
        <div class="card">
            <h3>🎤 الأوامر الصوتية/النصية</h3>
            <input type="text" id="commandInput" placeholder="اكتب أمر مثل: 'ابحث عن BTC' أو 'شغل التداول'" style="width: 70%;">
            <button class="btn" onclick="sendCommand()">🚀 تنفيذ الأمر</button>
            <div id="commandResult" style="margin-top: 10px; padding: 10px; background: #2a2e35; border-radius: 5px;"></div>
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
                    <label>نسبة المخاطرة %:</label>
                    <input type="number" name="RISK_PCT" value="{{ config.RISK_PCT * 100 }}" step="0.1" min="0.1" max="10">
                </div>
                <div class="form-group">
                    <label>التنفيذ التلقائي:</label>
                    <select name="AUTO_EXECUTE">
                        <option value="false" {% if not config.AUTO_EXECUTE %}selected{% endif %}>محاكاة</option>
                        <option value="true" {% if config.AUTO_EXECUTE %}selected{% endif %}>حقيقي</option>
                    </select>
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

        function getBalance() {
            fetch('/balance').then(r => r.json()).then(data => {
                alert('الرصيد: ' + JSON.stringify(data, null, 2));
            });
        }

        // الأوامر الصوتية/النصية
        function sendCommand() {
            const command = document.getElementById('commandInput').value;
            fetch('/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: command})
            }).then(r => r.json()).then(data => {
                document.getElementById('commandResult').innerHTML = data.response.replace(/\\n/g, '<br>');
            });
        }

        // إرسال الإعدادات
        document.getElementById('configForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            const data = Object.fromEntries(formData);
            
            // تحويل RISK_PCT إلى decimal
            if (data.RISK_PCT) {
                data.RISK_PCT = parseFloat(data.RISK_PCT) / 100;
            }
            
            fetch('/config', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            }).then(r => r.json()).then(data => {
                alert(data.message);
            });
        });

        // تحديث السجلات تلقائياً
        function updateLogs() {
            fetch('/logs').then(r => r.json()).then(data => {
                const logsContainer = document.getElementById('logs');
                logsContainer.innerHTML = data.logs.map(log => 
                    `<div>${log}</div>`
                ).join('');
            });
        }

        function clearLogs() {
            fetch('/clear_logs', {method: 'POST'}).then(() => updateLogs());
        }

        // تحديث كل 5 ثواني
        setInterval(updateStatus, 5000);
        setInterval(updateLogs, 5000);
        updateStatus();
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
    return jsonify({"running": trading_bot.is_running})

@app.route('/start', methods=['POST'])
def start_trading():
    if trading_bot.start_trading():
        return jsonify({"message": "تم بدء التداول التلقائي"})
    return jsonify({"message": "التداول مشغّل مسبقاً"})

@app.route('/stop', methods=['POST'])
def stop_trading():
    trading_bot.stop_trading()
    return jsonify({"message": "تم إيقاف التداول التلقائي"})

@app.route('/balance')
def balance():
    return jsonify(trading_bot.get_balance())

@app.route('/command', methods=['POST'])
def process_command():
    data = request.get_json()
    command = data.get('command', '')
    response = trading_bot.process_command(command)
    return jsonify({"response": response})

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
        "version": "2.0"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
