import language_tool_python

def is_grammatically_correct(text, language='pt-BR'):
    # Initialize LanguageTool with the Portuguese language model
    tool = language_tool_python.LanguageTool(language)

    # Check for grammar errors in the text
    matches = tool.check(text)

    #print(matches)

    # If there are no grammar errors, consider the text grammatically correct
    if len(matches) == 0:
        return True
    else:
        print("Erros gramaticais")
        for match in matches:
            print(match)
        return False

# Example usage
#text_to_check = "Este é ums exemplo de fraze em português sem erros de gramática."
text_to_check = "Mencionar as principais TPGs surgidas (ou desenvolvidas) ao longo da história revela a importância socioeconômicas dessas criações humanas. Não é exagero qualificá-las como revolucionárias, por contas das mudanças radicais e paradigmáticas operadas. Provavelmente, as principais TPGs observadas na sociedade humana são: a) eletricidade; b) internet e c) computador. No momento atual vai se formando um consenso de que a AGI (inteligência artificial geral) será incorporada nesse seleto grupo com potencial transformador ainda maior do que as anteriores."
result = is_grammatically_correct(text_to_check, language='pt-BR')
if result:
    print("The text is grammatically correct.")
else:
    print("The text contains grammar errors.")


