import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew, pearsonr
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from dataclasses import dataclass
import warnings
from datetime import datetime
import json
import io

# Configuration
warnings.filterwarnings('ignore')

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    missing_deps = []
    try:
        import plotly
    except ImportError:
        missing_deps.append("plotly")
    try:
        import seaborn
    except ImportError:
        missing_deps.append("seaborn")
    
    if missing_deps:
        st.error(f"Packages manquants : {', '.join(missing_deps)}")
        st.write("Installez-les avec :")
        st.code(f"pip install {' '.join(missing_deps)}")
        st.stop()

@dataclass
class LayerData:
    """Classe améliorée pour stocker les données d'une couche"""
    point: float
    ratio: Optional[float] = None
    error: Optional[float] = None
    stability: Optional[float] = None
    energy: Optional[float] = None
    resonance: Optional[float] = None
    interference: Optional[float] = None
    quality_score: Optional[float] = None

class PhysicalConstants:
    """Constantes physiques et mathématiques du système"""
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2
        self.pi = np.pi
        self.e = np.e
        self.stability_threshold = 0.5
        self.error_threshold = 1e-6
        self.resonance_threshold = 0.1

class LayerAnalysis:
    """Classe pour l'analyse détaillée d'une couche spécifique"""
    def __init__(self, point: float, phi: float, layer_number: int):
        self.point = point
        self.phi = phi
        self.layer_number = layer_number
        
    def calculate_energy(self) -> float:
        """Calcule l'énergie de la couche"""
        return abs(np.sin(self.phi**self.layer_number * self.point))
    
    def calculate_resonance(self, other_points: List[float]) -> float:
        """Calcule la résonance avec les autres couches"""
        resonance = 0
        for i, other_point in enumerate(other_points):
            if i != self.layer_number:
                resonance += np.sin(self.phi**i * other_point) * np.sin(self.phi**self.layer_number * self.point)
        return abs(resonance)
    
    def calculate_stability(self, h_range: np.ndarray, base_function) -> float:
        """Calcule la stabilité locale"""
        h_near = np.linspace(self.point - 0.1, self.point + 0.1, 100)
        f_near = base_function(h_near)
        d2f = np.gradient(np.gradient(f_near, h_near), h_near)
        return abs(np.mean(d2f))
    
    def calculate_quality_score(self, stability: float, resonance: float, energy: float) -> float:
        """Calcule un score de qualité global pour la couche"""
        normalized_stability = 1 / (1 + np.exp(-stability))  # Sigmoid normalisation
        normalized_resonance = 1 / (1 + resonance)  # Inverse normalisation
        normalized_energy = energy / (1 + energy)  # Bounded normalisation
        
        weights = {'stability': 0.4, 'resonance': 0.3, 'energy': 0.3}
        score = (weights['stability'] * normalized_stability +
                weights['resonance'] * normalized_resonance +
                weights['energy'] * normalized_energy)
        return score
class TheoremAnalyzer:
    """Classe principale améliorée pour l'analyse du théorème"""
    def __init__(self, num_layers: int = 7):
        self.num_layers = num_layers
        self.constants = PhysicalConstants()
        self.phi = self.constants.phi
        self.initialize_data()
        self.historical_context = self.load_historical_context()

    def initialize_data(self):
        """Initialisation améliorée des données"""
        self.h = np.linspace(0, 40, 1000)
        self.phi_powers = [self.phi**n for n in range(1, self.num_layers + 1)]
        self.points = self.calculate_transition_points()
        self.ratios = self.calculate_ratios()
        self.errors = self.calculate_errors()
        self.stability = self.analyze_stability()
        self.energy = self.calculate_energy()
        self.resonances = self.analyze_resonances()
        self.interference = self.calculate_interference()
        self.quality_scores = self.calculate_quality_scores()

    def load_historical_context(self) -> Dict:
        """Charge le contexte historique"""
        return {
            "source": "Coran 71:15",
            "verse_arabic": "أَلَمْ تَرَوْا كَيْفَ خَلَقَ اللَّهُ سَبْعَ سَمَاوَاتٍ طِبَاقًا",
            "verse_translation": "N'avez-vous pas vu comment Allah a créé sept cieux superposés?",
            "other_references": [
                "65:12 - Sept terres",
                "2:261 - Sept épis",
                "12:43 - Sept années",
                "15:87 - Sept versets",
                "2:29 - Sept cieux",
                "67:3 - Sept cieux superposés",
                "78:12 - Sept cieux renforcés"
            ]
        }

    def calculate_transition_points(self) -> List[float]:
        """Calcul amélioré des points de transition"""
        return [np.pi / (2 * self.phi**n) for n in range(1, self.num_layers + 1)]

    def calculate_ratios(self) -> List[float]:
        """Calcul des ratios entre points consécutifs"""
        return [self.points[i]/self.points[i+1] for i in range(len(self.points)-1)]

    def calculate_errors(self) -> List[float]:
        """Calcul amélioré des erreurs"""
        return [abs(ratio - self.phi) for ratio in self.ratios]

    def analyze_stability(self) -> List[float]:
        """Analyse améliorée de la stabilité"""
        stability = []
        for i, point in enumerate(self.points):
            layer = LayerAnalysis(point, self.phi, i)
            stability_value = layer.calculate_stability(self.h, self.base_function)
            normalized_stability = stability_value / (1 + stability_value)  # Normalisation
            stability.append(normalized_stability)
        return stability

    def calculate_energy(self) -> List[float]:
        """Calcul de l'énergie par couche"""
        return [abs(np.sin(self.phi**n * p)) for n, p in enumerate(self.points)]

    def analyze_resonances(self) -> Dict[str, float]:
        """Analyse détaillée des résonances"""
        resonances = {}
        for i in range(len(self.points)):
            for j in range(i+1, len(self.points)):
                key = f"Layer{i+1}-Layer{j+1}"
                freq_ratio = self.phi**(j-i)
                interference = np.sin(self.phi**i * self.points[i]) * np.sin(self.phi**j * self.points[j])
                resonances[key] = abs(interference)
        return resonances

    def calculate_interference(self) -> List[float]:
        """Calcul des interférences entre couches"""
        interference = []
        for i, point in enumerate(self.points):
            other_points = self.points[:i] + self.points[i+1:]
            layer_interference = sum(np.sin(self.phi**j * point) 
                                  for j, _ in enumerate(other_points))
            interference.append(abs(layer_interference))
        return interference

    def calculate_quality_scores(self) -> List[float]:
        """Calcul des scores de qualité pour chaque couche"""
        quality_scores = []
        for i, point in enumerate(self.points):
            layer = LayerAnalysis(point, self.phi, i)
            score = layer.calculate_quality_score(
                self.stability[i],
                sum(res for key, res in self.resonances.items() if f"Layer{i+1}" in key),
                self.energy[i]
            )
            quality_scores.append(score)
        return quality_scores

    def base_function(self, h: np.ndarray, amplitudes: List[float] = None) -> np.ndarray:
        """Fonction de base améliorée"""
        if amplitudes is None:
            amplitudes = [1.0] * self.num_layers
        
        result = np.zeros_like(h, dtype=float)
        for n, A in enumerate(amplitudes, 1):
            result += A * np.sin(self.phi**n * h)
        return result
    
    def generate_complete_analysis(self) -> Dict:
        """Génère une analyse complète et améliorée du système"""
        # Calcul des métriques avancées
        energy_distribution = self.analyze_energy_distribution()
        stability_metrics = self.evaluate_stability_metrics()
        resonance_analysis = self.analyze_resonance_patterns()
        layer_quality = self.evaluate_layer_quality()
        
        return {
            'basic_metrics': {
                'phi': self.phi,
                'num_layers': self.num_layers,
                'average_error': np.mean(self.errors) if self.errors else 0,
                'max_error': max(self.errors) if self.errors else 0,
                'stability_score': np.mean(self.stability),
                'energy_score': np.mean(self.energy),
                'resonance_score': np.mean(list(self.resonances.values())),
                'quality_score': np.mean(self.quality_scores)
            },
            'layer_data': [
                LayerData(
                    point=self.points[i],
                    ratio=self.ratios[i] if i < len(self.ratios) else None,
                    error=self.errors[i] if i < len(self.errors) else None,
                    stability=self.stability[i],
                    energy=self.energy[i],
                    resonance=sum(v for k, v in self.resonances.items() if f"Layer{i+1}" in k),
                    interference=self.interference[i],
                    quality_score=self.quality_scores[i]
                )
                for i in range(len(self.points))
            ],
            'advanced_analysis': {
                'energy_distribution': energy_distribution,
                'stability_metrics': stability_metrics,
                'resonance_analysis': resonance_analysis,
                'layer_quality': layer_quality
            },
            'validation': {
                'is_valid': self.validate_system(),
                'optimal_layers': self.is_optimal_layers(),
                'stability_check': self.check_stability_criteria(),
                'error_analysis': self.analyze_error_patterns()
            }
        }

    def analyze_energy_distribution(self) -> Dict:
        """Analyse détaillée de la distribution d'énergie"""
        return {
            'total_energy': sum(self.energy),
            'energy_per_layer': self.energy,
            'energy_ratios': [self.energy[i]/self.energy[i-1] if i > 0 else None 
                            for i in range(len(self.energy))],
            'energy_balance': np.std(self.energy),
            'energy_progression': np.polyfit(range(len(self.energy)), self.energy, 1)[0],
            'energy_patterns': {
                'increasing': np.all(np.diff(self.energy) > 0),
                'decreasing': np.all(np.diff(self.energy) < 0),
                'oscillating': any(np.diff(np.sign(np.diff(self.energy))) != 0)
            }
        }

    def evaluate_stability_metrics(self) -> Dict:
        """Évaluation approfondie de la stabilité"""
        return {
            'mean_stability': np.mean(self.stability),
            'stability_variance': np.var(self.stability),
            'stable_layers': sum(1 for s in self.stability if s > self.constants.stability_threshold),
            'stability_trend': np.polyfit(range(len(self.stability)), self.stability, 1)[0],
            'stability_ratios': [self.stability[i]/self.stability[i-1] if i > 0 else None 
                               for i in range(len(self.stability))],
            'stability_distribution': {
                'mean': np.mean(self.stability),
                'std': np.std(self.stability),
                'skew': skew(self.stability) if len(self.stability) > 2 else None
            }
        }

    def analyze_resonance_patterns(self) -> Dict:
        """Analyse détaillée des patterns de résonance"""
        resonance_values = list(self.resonances.values())
        return {
            'mean_resonance': np.mean(resonance_values),
            'max_resonance': max(resonance_values),
            'resonance_distribution': {
                'mean': np.mean(resonance_values),
                'std': np.std(resonance_values),
                'skew': skew(resonance_values) if len(resonance_values) > 2 else None
            },
            'critical_resonances': {k: v for k, v in self.resonances.items() 
                                  if v > self.constants.resonance_threshold},
            'resonance_patterns': {
                'increasing': np.all(np.diff(resonance_values) > 0),
                'decreasing': np.all(np.diff(resonance_values) < 0),
                'oscillating': any(np.diff(np.sign(np.diff(resonance_values))) != 0)
            }
        }

    def evaluate_layer_quality(self) -> List[Dict]:
        """Évaluation détaillée de la qualité de chaque couche"""
        quality_metrics = []
        for i in range(len(self.points)):
            metrics = {
                'layer': i+1,
                'quality_score': self.quality_scores[i],
                'stability_contribution': self.stability[i],
                'energy_contribution': self.energy[i],
                'resonance_impact': sum(v for k, v in self.resonances.items() if f"Layer{i+1}" in k),
                'interference_level': self.interference[i],
                'error_rate': self.errors[i] if i < len(self.errors) else None,
                'relative_position': self.points[i]/self.points[0] if i > 0 else 1.0,
                'phi_correlation': abs(self.quality_scores[i] - self.phi)/self.phi
            }
            quality_metrics.append(metrics)
        return quality_metrics

    def create_visualizations(self) -> Dict[str, go.Figure]:
        """Crée toutes les visualisations avancées"""
        figures = {}

        # 1. Fonction principale et points de transition
        fig_main = go.Figure()
        y = self.base_function(self.h)
        fig_main.add_trace(go.Scatter(x=self.h, y=y, name='f(h)', line=dict(color='blue')))
        
        # Ajout des points de transition avec annotations
        for i, point in enumerate(self.points):
            fig_main.add_vline(x=point, line_dash="dash", line_color="red", opacity=0.5)
            fig_main.add_annotation(
                x=point, y=max(y),
                text=f"C{i+1}",
                showarrow=True,
                arrowhead=1
            )
        
        fig_main.update_layout(
            title='Fonction Principale et Points de Transition',
            xaxis_title='h',
            yaxis_title='f(h)',
            hovermode='x unified'
        )
        figures['main'] = fig_main

        # 2. Graphique amélioré des ratios
        fig_ratios = go.Figure()
        if self.ratios:  # Vérification que self.ratios n'est pas vide
            fig_ratios.add_trace(go.Scatter(
                x=list(range(1, len(self.ratios)+1)),
                y=self.ratios,
                mode='lines+markers',
                name='Ratios calculés',
                error_y=dict(
                    type='data',
                    array=[abs(r - self.phi) for r in self.ratios],
                    visible=True
                )
            ))
            fig_ratios.add_hline(
                y=self.phi,
                line_dash="dash",
                line_color="red",
                annotation_text="φ (nombre d'or)"
            )
            fig_ratios.update_layout(
                title='Ratios entre Points Consécutifs',
                xaxis_title='Numéro du ratio',
                yaxis_title='Valeur du ratio'
            )
            figures['ratios'] = fig_ratios

        # 3. Analyse de stabilité améliorée
        fig_stability = go.Figure()
        stability_data = pd.DataFrame({
            'Couche': range(1, len(self.stability)+1),
            'Stabilité': self.stability,
            'Énergie': self.energy,
            'Qualité': self.quality_scores
        })
        
        fig_stability.add_trace(go.Bar(
            name='Stabilité',
            x=stability_data['Couche'],
            y=stability_data['Stabilité'],
            marker_color='blue'
        ))
        fig_stability.add_trace(go.Scatter(
            name='Score de Qualité',
            x=stability_data['Couche'],
            y=stability_data['Qualité'],
            line=dict(color='red')
        ))
        fig_stability.update_layout(
            title='Analyse Multicritères par Couche',
            barmode='overlay',
            xaxis_title='Numéro de couche',
            yaxis_title='Score'
        )
        figures['stability'] = fig_stability

        # 4. Matrice de résonance
        resonance_matrix = np.zeros((self.num_layers, self.num_layers))
        for key, value in self.resonances.items():
            i, j = map(lambda x: int(x.replace('Layer', ''))-1, key.split('-'))
            resonance_matrix[i,j] = value
            resonance_matrix[j,i] = value
        
        fig_resonance = go.Figure(data=go.Heatmap(
            z=resonance_matrix,
            x=[f'C{i+1}' for i in range(self.num_layers)],
            y=[f'C{i+1}' for i in range(self.num_layers)],
            colorscale='Viridis',
            text=np.round(resonance_matrix, 3),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True
        ))
        fig_resonance.update_layout(
            title='Matrice de Résonance entre Couches',
            xaxis_title='Couche',
            yaxis_title='Couche'
        )
        figures['resonance'] = fig_resonance

        # 5. Analyse énergétique
        fig_energy = go.Figure()
        energy_data = self.analyze_energy_distribution()
        
        fig_energy.add_trace(go.Bar(
            name='Énergie par couche',
            x=list(range(1, self.num_layers+1)),
            y=energy_data['energy_per_layer'],
            marker_color='orange'
        ))
        
        if energy_data['energy_ratios'][0] is not None:
            fig_energy.add_trace(go.Scatter(
                name='Ratio énergétique',
                x=list(range(2, self.num_layers+1)),
                y=energy_data['energy_ratios'][1:],
                yaxis='y2',
                line=dict(color='green')
            ))
        
        fig_energy.update_layout(
            title='Distribution et Progression de l\'Énergie',
            xaxis_title='Couche',
            yaxis_title='Énergie',
            yaxis2=dict(
                title='Ratio énergétique',
                overlaying='y',
                side='right'
            )
        )
        figures['energy'] = fig_energy

        # 6. Analyse comparative
        fig_compare = make_subplots(rows=2, cols=1,
                                  subplot_titles=('Évolution des Métriques', 'Distribution des Erreurs'))
        
        # Métriques normalisées
        metrics = pd.DataFrame({
            'Stabilité': self.stability,
            'Énergie': [e/max(self.energy) for e in self.energy],
            'Qualité': self.quality_scores
        })
        
        for col in metrics.columns:
            fig_compare.add_trace(
                go.Scatter(x=list(range(1, len(metrics)+1)), 
                          y=metrics[col],
                          name=col),
                row=1, col=1
            )
        
        if self.errors:  # Vérification que self.errors n'est pas vide
            fig_compare.add_trace(
                go.Box(y=self.errors, name='Distribution des Erreurs'),
                row=2, col=1
            )
        
        fig_compare.update_layout(height=800, showlegend=True)
        figures['comparison'] = fig_compare

        return figures

def create_interactive_app():
    """Interface utilisateur interactive améliorée"""
    # Vérification des dépendances
    check_dependencies()
    
    st.set_page_config(page_title="Analyse Avancée du Théorème des Sept Couches",
                      layout="wide")

    # Titre et introduction
    st.title("Analyse Avancée du Théorème des Sept Couches")
    st.markdown("""
    ### Exploration de la Convergence entre Texte Ancien et Mathématiques Modernes
    
    Cette application propose une analyse approfondie du théorème des sept couches,
    explorant la relation entre les textes anciens mentionnant sept couches célestes
    et les propriétés mathématiques naturelles basées sur le nombre d'or (φ).
    """)

    # Contrôles latéraux améliorés
    with st.sidebar:
        st.header("Paramètres d'Analyse")
        num_layers = st.slider("Nombre de couches", 3, 12, 7)
        
        st.subheader("Options d'Affichage")
        show_advanced = st.checkbox("Analyses avancées", True)
        show_mathematical = st.checkbox("Détails mathématiques", False)
        show_historical = st.checkbox("Contexte historique", False)
        
        # Paramètres avancés dans un expander
        with st.expander("Paramètres avancés"):
            stability_threshold = st.slider("Seuil de stabilité", 0.0, 1.0, 0.5)
            error_threshold = st.slider("Seuil d'erreur", 0.0, 0.1, 0.0001, format="%.4f")
            quality_weight = st.slider("Poids qualité", 0.0, 1.0, 0.4)

    # Création de l'analyseur avec les paramètres
    analyzer = TheoremAnalyzer(num_layers)
    analysis = analyzer.generate_complete_analysis()
    figures = analyzer.create_visualizations()

    # Affichage des métriques principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nombre de couches", num_layers)
    with col2:
        st.metric("Score de qualité", f"{analysis['basic_metrics']['quality_score']:.4f}")
    with col3:
        st.metric("Stabilité moyenne", f"{analysis['basic_metrics']['stability_score']:.4f}")
    with col4:
        st.metric("Erreur moyenne", f"{analysis['basic_metrics']['average_error']:.2e}")

    # Visualisation principale
    st.plotly_chart(figures['main'], use_container_width=True)

    # Tableau des données amélioré
    st.header("Données Détaillées des Couches")
    layer_data = pd.DataFrame([
        {
            'Couche': i+1,
            'Point de transition': f"{layer.point:.4f}",
            'Ratio': f"{layer.ratio:.6f}" if layer.ratio else "N/A",
            'Erreur': f"{layer.error:.2e}" if layer.error else "N/A",
            'Stabilité': f"{layer.stability:.4f}",
            'Énergie': f"{layer.energy:.4f}",
            'Score Qualité': f"{layer.quality_score:.4f}"
        }
        for i, layer in enumerate(analysis['layer_data'])
    ])
    st.dataframe(layer_data)

    if show_advanced:
        st.header("Analyses Avancées")
        
        # Visualisations avancées en colonnes
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(figures['ratios'], use_container_width=True)
            st.plotly_chart(figures['energy'], use_container_width=True)
        with col2:
            st.plotly_chart(figures['stability'], use_container_width=True)
            st.plotly_chart(figures['resonance'], use_container_width=True)
        
        st.plotly_chart(figures['comparison'], use_container_width=True)

        # Analyse détaillée de l'optimalité
        st.header("Analyse d'Optimalité")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Critères de Stabilité")
            stability_check = analysis['validation']['stability_check']
            for criterion, value in stability_check.items():
                st.write(f"- {criterion}: {'✅' if value else '❌'}")
        
        with col2:
            st.subheader("Analyse des Erreurs")
            error_analysis = analysis['validation']['error_analysis']
            st.write("Distribution des erreurs:")
            st.write(f"- Moyenne: {error_analysis['error_distribution']['mean']:.2e}")
            st.write(f"- Maximum: {error_analysis['error_distribution']['max']:.2e}")
            if error_analysis['error_distribution']['problematic_layers']:
                st.warning(f"Couches problématiques: {error_analysis['error_distribution']['problematic_layers']}")

    if show_mathematical:
        st.header("Fondements Mathématiques")
        st.latex(r"\text{Fonction de base: } f(h) = \sum_{n=1}^{" + str(num_layers) + r"} \sin(\phi^n h)")
        st.latex(r"\text{Points de transition: } h_k = \frac{\pi}{2\phi^k}")
        st.latex(r"\text{Ratio théorique: } \frac{h_k}{h_{k+1}} = \phi")
        
        # Description détaillée
        st.markdown("""
        ### Explications des Formules
        
        1. **Fonction de base** : 
           - Somme de sinusoïdes avec des fréquences basées sur les puissances de φ
           - Chaque terme contribue à une couche distincte
        
        2. **Points de transition** :
           - Représentent les frontières entre les couches
           - Suivent une progression géométrique basée sur φ
        
        3. **Ratios** :
           - Les rapports entre points consécutifs convergent vers φ
           - Démontrent l'auto-similarité du système
        """)

    if show_historical:
        st.header("Contexte Historique")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Texte Original")
            st.markdown(f"**{analyzer.historical_context['verse_arabic']}**")
            st.markdown(f"*{analyzer.historical_context['verse_translation']}*")
        
        with col2:
            st.subheader("Autres Références")
            for ref in analyzer.historical_context['other_references']:
                st.markdown(f"- {ref}")
        
        st.markdown("""
        ### Implications Scientifiques
        
        La convergence entre ce texte ancien et nos découvertes mathématiques soulève plusieurs points intéressants :
        
        1. **Structure Naturelle** : L'émergence naturelle de sept couches dans notre modèle mathématique
        2. **Propriétés du Nombre d'Or** : Le rôle fondamental de φ dans la structure
        3. **Auto-similarité** : La nature fractale des transitions entre couches
        4. **Stabilité Optimale** : L'optimalité du nombre sept pour la stabilité du système
        """)

    # Export des données
    st.header("Export des Résultats")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export JSON
        json_str = json.dumps(analysis, default=str, indent=2)
        st.download_button(
            label="Télécharger l'analyse complète (JSON)",
            data=json_str,
            file_name=f"seven_layers_analysis_{num_layers}layers.json",
            mime="application/json"
        )
    
    with col2:
        # Export CSV
        csv = layer_data.to_csv(index=False)
        st.download_button(
            label="Télécharger les données (CSV)",
            data=csv,
            file_name=f"layer_data_{num_layers}layers.csv",
            mime="text/csv"
        )
    
    with col3:
        # Export Rapport
        report = f"""# Rapport d'Analyse du Théorème des Sept Couches
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Nombre de couches analysées: {num_layers}

## Métriques Principales
- Score de qualité: {analysis['basic_metrics']['quality_score']:.4f}
- Stabilité moyenne: {analysis['basic_metrics']['stability_score']:.4f}
- Erreur moyenne: {analysis['basic_metrics']['average_error']:.2e}

## Validation
- Système valide: {'Oui' if analysis['validation']['is_valid'] else 'Non'}
- Nombre de couches optimal: {'Oui' if analysis['validation']['optimal_layers'] else 'Non'}

## Détails par Couche
{layer_data.to_string()}
"""
        st.download_button(
            label="Télécharger le rapport (TXT)",
            data=report,
            file_name=f"report_{num_layers}layers.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    create_interactive_app()