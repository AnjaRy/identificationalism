#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Programmiertechniken in der Computerlinguistik II
# Übung 1 - Aufgabe 2
# Datum: 06.03.2018
# Name: Vera Bernhard (17729583) & Anja Ryser (17704461)

def join_lines(sep_stream, dehyph_stream=None):
    """Formats a text into a more readable form with one line break between a 
        paragraph.

    Parameter:
        sep_stream: the name of file (string) or a string (in a list) 
        to be formatted
        dehyph_stream: the name of file (string) to write the converted text in

    Returns: A file with a more readable text"""

    # Check if sep_stream is a file name or already the string.
    if type(sep_stream) is str:
        open_file = open(sep_stream, 'r')
    else:
        open_file = sep_stream

    with open(dehyph_stream, 'w') as file_w:
        previous_c = ''
        for line in open_file:
            if line is '\n':
                file_w.write('\n')
                
            else:
                for char in line:
                    if char == '.':
                        if previous_c != previous_c.upper():
                            file_w.write(char + '\n')
                        else: 
                            file_w.write(char)
                            

                    elif char == ' ':
                        if previous_c == '.':
                            pass
                        elif previous_c == ' ':
                            pass
                        elif previous_c == '\n':
                            pass
                        else:
                            file_w.write(char)
                    
                    elif char == '\n':
                        if previous_c != '.':
                            file_w.write(' ')
                        else: 
                            pass
                        
                    else: 
                        file_w.write(char)
                        
                    previous_c = char
   
    if type(open_file) is not list:
      open_file.close()
      
      
def main():
    join_lines("Edgar Allan Poe.txt", "Preprocessed.txt")    
                        
if __name__ == '__main__':
    main()
    
    
#Anmerkungen: 
#Die Gedichte haben eine sehr komische Form angenommen, da es aber nicht so einfach ging, das zu verbessern, liess ich es so.
    
    
    
#anführungszeichen und klammer nach punkt
#eingerückter dialog
#linie startet mit leerzeichen
#gruusig: ca zeile 1300
#...
#mehrere Leerzeichen