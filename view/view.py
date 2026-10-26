import base64
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from controller.annotations import normalizar_anotacion
from model.model import ObjetoModel

BASE_DIR = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "view" / "templates"))
objetos = ObjetoModel()
router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def inicio(request: Request, q: str = ""):
	return templates.TemplateResponse(
		request=request,
		name="inicio.html",
		context={},
	)


@router.get("/buscar", response_class=HTMLResponse)
async def buscar(
	request: Request,
	q: str = "",
	campo: str = "todos",
	orden: str = "recientes",
):
	return templates.TemplateResponse(
		request=request,
		name="buscar.html",
		context={
			"objetos": objetos.buscar(q, campo, orden) if q.strip() else [],
			"consulta": q,
			"campo": campo,
			"orden": orden,
		},
	)


@router.get("/biblioteca", response_class=HTMLResponse)
async def biblioteca(request: Request):
	return templates.TemplateResponse(
		request=request,
		name="biblioteca.html",
		context={"objetos": objetos.listar()},
	)


@router.get("/nueva", response_class=HTMLResponse)
async def nueva_medicion(request: Request):
	return templates.TemplateResponse(request=request, name="nueva.html")


@router.post("/nueva")
async def guardar_medicion(
	 nombre: str = Form(...),
	 referencia: str = Form(""),
	 descripcion: str = Form(""),
	 imagen: UploadFile = File(...),
):
	datos = await imagen.read()
	if not datos:
		return RedirectResponse("/nueva?error=empty", status_code=303)

	try:
		objeto_id = objetos.procesar_y_guardar(
			nombre, referencia, descripcion, imagen.filename or "objeto", datos
		)
	except ValueError:
		return RedirectResponse("/nueva?error=measure", status_code=303)

	return RedirectResponse(f"/objetos/{objeto_id}", status_code=303)


@router.get("/objetos/{objeto_id}", response_class=HTMLResponse)
async def detalle(request: Request, objeto_id: int):
	objeto = objetos.obtener(objeto_id)
	if objeto is None:
		return RedirectResponse("/", status_code=303)
	objetos.completar_medidas(objeto_id)
	objeto = objetos.obtener(objeto_id)

	return templates.TemplateResponse(
		request=request,
		name="detalle.html",
		context={"objeto": objeto},
	)

@router.get("/objetos/{objeto_id}/modificar", response_class=HTMLResponse)
async def modificar_formulario(request: Request, objeto_id: int):
	objeto = objetos.obtener(objeto_id)
	if objeto is None:
		return RedirectResponse("/", status_code=303)
	return templates.TemplateResponse(
		request=request, name="modificar.html", context={"objeto": objeto}
	)


@router.post("/objetos/{objeto_id}/modificar")
async def modificar_objeto(
	objeto_id: int,
	nombre: str = Form(...),
	referencia: str = Form(""),
	descripcion: str = Form(""),
	imagen: UploadFile | None = File(None),
):
	datos = await imagen.read() if imagen is not None else None
	try:
		objetos.actualizar(
			objeto_id, nombre, referencia, descripcion,
			imagen.filename if imagen else None, datos,
		)
	except ValueError:
		return RedirectResponse(
			f"/objetos/{objeto_id}/modificar?error=measure", status_code=303
		)
	return RedirectResponse(f"/objetos/{objeto_id}", status_code=303)


@router.post("/objetos/{objeto_id}/eliminar")
async def eliminar_objeto(objeto_id: int):
	objetos.eliminar(objeto_id)
	return RedirectResponse("/", status_code=303)


@router.post("/objetos/{objeto_id}/anotaciones")
async def guardar_anotaciones(objeto_id: int, imagen: str = Form(...)):
	try:
		contenido = imagen.split(",", 1)[1]
		png = base64.b64decode(contenido)
		objetos.guardar_anotaciones(objeto_id, normalizar_anotacion(png))
	except (ValueError, IndexError, base64.binascii.Error):
		return Response(status_code=400)
	return {"ok": True}


@router.get("/objetos/{objeto_id}/imagen/{tipo}")
async def imagen(objeto_id: int, tipo: str):
	columnas = {
		"original": "imagen_original",
		"color": "imagen_color",
		"bordes": "imagen_bordes",
		"anotaciones": "anotaciones",
		"a4": "imagen_a4",
		"folio": "imagen_folio",
		"mascara": "imagen_mascara",
		"resultado": "imagen_resultado",
	}
	columna = columnas.get(tipo)
	objeto = objetos.obtener(objeto_id)
	color_recalculado = objetos.obtener_color(objeto_id) if tipo == "color" else None

	if columna is None or objeto is None:
		return Response(status_code=404)
	contenido = color_recalculado or objeto[columna]
	if contenido is None:
		return Response(status_code=404)

	media_type = "image/png" if tipo in {"mascara", "color", "bordes", "anotaciones"} else "image/jpeg"
	return Response(content=contenido, media_type=media_type)


def configure_views(app):
	app.include_router(router)
	app.mount(
		"/static",
		StaticFiles(directory=str(BASE_DIR / "view" / "static")),
		name="static",
	)
