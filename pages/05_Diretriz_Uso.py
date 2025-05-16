import streamlit as st

st.title("Diretriz de Uso e Conclusões")

st.header("1. Propósito do Modelo")
st.markdown("""
O modelo desenvolvido tem como principal objetivo **oferecer um apoio analítico ao processo seletivo**, a partir de dados textuais extraídos de currículos e descrições de vagas. A proposta é **fornecer insights baseados em padrões históricos de contratação**, não substituir a avaliação humana.

A ideia central é que a inteligência artificial funcione como um **sistema de recomendação**: 
indicando candidatos com maior similaridade textual com perfis aprovados anteriormente, e alertando para prováveis desconexões. 
""")

st.header("2. Limitações Identificadas")
st.markdown("""
Apesar da sofisticação da abordagem (TF-IDF, SVD, XGBoost otimizado e integração de dados), o modelo apresenta **limitações relevantes que restringem seu uso como ferramenta decisória definitiva**:

- **F1-score da classe 1** (contratação) gira em torno de **0.55**, com recall em **0.56**;
- O **valor da AUC é 0.719**, o que indica boa discriminatividade, mas não excelente;
- Houve ocorrências de **palavras-chave compartilhadas** entre candidatos aprovados e reprovados, dificultando a distinção com base textual pura;
- Alguns aspectos da decisão de contratação estão **fora do escopo dos dados coletados**: entrevistas, dinâmica comportamental, preferência de gestores, timing, etc.

> A previsão feita pelo modelo é limitada às features presentes, e não deve ser interpretada como juízo determinístico.
""")

st.header("3. Papel Recomendado para o Modelo")
st.markdown("""
Diante do que foi apresentado, o uso sugerido para esta solução é o de **ferramenta de suporte ao RH**:

- **Priorizar** candidatos com alta similaridade textual com histórico de aprovação;
- **Identificar lacunas potenciais** em perfis com baixa confiança;
- Auxiliar em **triagens iniciais** em bases amplas de candidatos;
- **Fornecer insights quantitativos** sobre características mais frequentes entre aprovados;
- **Ancorar decisões preliminares com base em histórico**, mas sem suprimir a etapa qualitativa do processo.

> A decisão final deve permanecer sob responsabilidade humana, com o modelo atuando como aliado estratégico.
""")

st.header("4. Considerações Finais")
st.markdown("""
Este projeto demonstra o potencial do uso de IA aplicada ao recrutamento, mas também reforça a importância da interpretação crítica. Nenhum modelo é capaz de captar a totalidade dos fatores que levam à contratação.

O modelo atual oferece:
- Robustez técnica com pipeline estruturado
- Resultados consistentes para classificação textual
- Transparência sobre métricas e processo de construção

Mas também requer:
- Uso contextualizado
- Complementação com avaliação qualitativa
- Monitoramento contínuo e atualização dos dados

> Em síntese, trata-se de um sistema de apoio inteligente à tomada de decisão, não de substituição.
""")