from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

def generar_grafica_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        target = request.form.get('target')
        
        if not file or not target:
            return render_template('index.html', error="Sube un archivo y escribe la variable objetivo.")
        
        try:
            df = pd.read_csv(file).dropna()
            tabla_html = df.head().to_html(classes='striped', index=False)
            
            if target not in df.columns:
                return render_template('index.html', error=f"La columna '{target}' no existe en tu CSV.")
                
            if df[target].dtype == 'object':
                df[target] = df[target].astype('category').cat.codes
                
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target in num_cols: 
                num_cols.remove(target)
            
            if not num_cols:
                return render_template('index.html', error="No hay columnas numéricas para analizar.")
                
            evidencia_col = num_cols[0]
            umbral = df[evidencia_col].median()
            
            A = df[target] == 1
            B = df[evidencia_col] > umbral
            
            p_fallo = A.mean() 
            p_evidencia = B.mean() 
            p_evidencia_dado_fallo = B[A].mean() if A.sum() > 0 else 0
            
            if p_evidencia > 0:
                p_fallo_dado_evidencia = (p_evidencia_dado_fallo * p_fallo) / p_evidencia
            else:
                p_fallo_dado_evidencia = 0
            
            X = df[num_cols]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            model = GaussianNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.style.use('dark_background') 
            
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='mako')
            plt.title("Matriz de Confusión")
            plt.xlabel("Predicción del Modelo")
            plt.ylabel("Dato Real")
            grafica_cm = generar_grafica_base64()
            
            plt.figure(figsize=(5,4))
            plt.bar(['P(Fallo)\nGeneral', f'P(Fallo)\nsi {evidencia_col} > {umbral:.1f}'], 
                    [p_fallo, p_fallo_dado_evidencia], 
                    color=['#4f5b66', '#39c5bb'])
            plt.title("Impacto de la Evidencia (Bayes)")
            plt.ylim(0, 1)
            plt.ylabel("Probabilidad")
            grafica_bayes = generar_grafica_base64()
            
            plt.style.use('default')
            
            resultados = {
                'tabla_datos': tabla_html,
                'p_fallo': round(p_fallo, 4),
                'p_evidencia': round(p_evidencia, 4),
                'p_evidencia_dado_fallo': round(p_evidencia_dado_fallo, 4),
                'p_fallo_dado_evidencia': round(p_fallo_dado_evidencia, 4),
                'evidencia_col': evidencia_col,
                'umbral': round(umbral, 2),
                'acc': round(acc, 4),
                'grafica_cm': grafica_cm,
                'grafica_bayes': grafica_bayes
            }
            
            return render_template('index.html', res=resultados)
            
        except Exception as e:
            return render_template('index.html', error=f"Hubo un error al procesar los datos: {str(e)}")
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        port=443, 
        ssl_context=(
            '/etc/letsencrypt/live/hatsunemikugod.ddns.net/fullchain.pem', 
            '/etc/letsencrypt/live/hatsunemikugod.ddns.net/privkey.pem'
        )
    )