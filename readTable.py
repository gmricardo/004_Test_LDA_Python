import openpyxl

def leer_columna_xlsx(nombre_archivo, nombre_columna):
    wb = openpyxl.load_workbook(nombre_archivo)
    hoja = wb.active

    columna = hoja[nombre_columna]
    valores = [celda.value for celda in columna]

    return valores

# Ejemplo de uso
nombre_archivo = 'incidentes_cliente_360.xlsx'
nombre_columna = 'A'  # Letra de la columna que deseas leer

info_clientes_360 = leer_columna_xlsx(nombre_archivo, nombre_columna)


