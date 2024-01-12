from pygame import display
from pygame import draw
from pygame import font
from pygame import quit
from pygame import event
from pygame import QUIT
from pygame import MOUSEBUTTONDOWN
from pygame import Surface

def init_graphics():
    global S_WIDTH
    S_WIDTH = 1000
    global S_HEIGHT
    S_HEIGHT = 1000
    global NODE_RADIUS
    NODE_RADIUS = 5
    display.init()
    font.init()
    global NODE_FONT
    NODE_FONT = font.Font('freesansbold.ttf', 20)
    global TITLE_FONT
    TITLE_FONT = font.Font('freesansbold.ttf', 40)
    global screen
    screen = display.set_mode((S_WIDTH, S_HEIGHT))
    global saved_surfaces
    saved_surfaces = {}

ENABLE_GRAPHICS = True
if ENABLE_GRAPHICS:
    init_graphics()
else:
    screen = None

def get_saved_surface(identifier):
    if identifier == None:
        surface = screen
    else:
        if not identifier in saved_surfaces:
            saved_surfaces[identifier] = Surface((S_WIDTH,S_HEIGHT))
        surface = saved_surfaces[identifier]
    return surface

def save_surface(saved_surface,surface_identifier):
    if surface_identifier == None:
        return
    else:
        saved_surfaces[surface_identifier] = saved_surface

def draw_line(a,b,color,surface_identifier = None,width = 1):
    srfc = get_saved_surface(surface_identifier)
    draw.line(srfc,color,(a.getx()*10, a.gety()*10),(b.getx()*10,b.gety()*10),width)
    draw.circle(srfc,(255,255,255),(a.getx()*10,a.gety()*10),NODE_RADIUS)
    draw.circle(srfc,(255,255,255),(b.getx()*10,b.gety()*10),NODE_RADIUS)
    save_surface(srfc,surface_identifier)

def draw_circuit(path,color, surface_identifier = None,width = 1):#Take pathCircuit iterator
    if not ENABLE_GRAPHICS:
        return
    srfc = get_saved_surface(surface_identifier)
    previous = next(path)
    for current in path:
        draw.circle(srfc,(255,255,255),(previous.getx()*10,previous.gety()*10),NODE_RADIUS)
        draw.line(srfc,color,(previous.getx()*10, previous.gety()*10),(current.getx()*10,current.gety()*10),width)
        text = NODE_FONT.render(str(previous.id), True, (255,255,255))
        textRect = text.get_rect()
        textRect.center = (previous.getx()*10+10,previous.gety()*10-10)
        srfc.blit(text,textRect)
        previous = current
    save_surface(srfc,surface_identifier)

def draw_AB_circuit(path,color1,color2,width = 1,surface_identifier = screen):#Take pathCircuit iterator
    srfc = get_saved_surface(surface_identifier)
    previous = next(path)
    for current in path:
        if current.A == True:
            color = color1
            width = 1
        else:
            color = color2
            width = 2
        draw.circle(srfc,(255,255,255),(previous.getx()*10,previous.gety()*10),NODE_RADIUS)
        draw.line(srfc,color,(previous.getx()*10, previous.gety()*10),(current.getx()*10,current.gety()*10),width)
        text = NODE_FONT.render(str(previous.id), True, (255,255,255))
        textRect = text.get_rect()
        textRect.center = (previous.getx()*10+10,previous.gety()*10-10)
        srfc.blit(text,textRect)
        previous = current
    save_surface(srfc,surface_identifier)

def phase_title(title,surface_identifier = screen):
    srfc = get_saved_surface(surface_identifier)
    text = TITLE_FONT.render(title, True, (255,255,255))
    textRect = text.get_rect()
    textRect.center = (500,900)
    srfc.blit(text,textRect)
    save_surface(srfc,surface_identifier)

def wait_for_click():
    while True:
        for evnt in event.get():
            if evnt.type == QUIT:
                quit()
            if evnt.type == MOUSEBUTTONDOWN:
                display.update()
                return    

def update():
    display.update()
def clear(surface_identifier = screen):
    if not ENABLE_GRAPHICS:
        return
    srfc = get_saved_surface(surface_identifier)
    srfc.fill((0,0,0))
    save_surface(srfc,surface_identifier)

def cycle_cover(PathA_iterator,PathB_iterator,cycles,edgeList_iterator_class,save_to=None):
    if not ENABLE_GRAPHICS:
        return
    draw_circuit(PathA_iterator,(255,0,0),save_to,1)
    draw_circuit(PathB_iterator,(255,0,0),save_to,1)
    for cycle in cycles:
        draw_circuit(edgeList_iterator_class(cycle),(0,255,0,),save_to,2)

def display_saved_surface(surface_identifier,overlap = False):
    if not ENABLE_GRAPHICS:
        return
    if not saved_surfaces[surface_identifier]:
        raise Exception('Surface: "{}" does not exist'.format(surface_identifier))  
    screen.blit(saved_surfaces[surface_identifier],(0,0))
    saved_surfaces[surface_identifier].fill((0,0,0))
    display.update()
    wait_for_click()

def draw_paths(paths,pathiterator_class,surface_identifier):
    if not ENABLE_GRAPHICS:
        return
    for path in paths:
        draw_circuit(pathiterator_class(path),(255,255,0),surface_identifier)

def draw_path(path,surface_identifier):
    if not ENABLE_GRAPHICS:
        return
    draw_circuit(path,(0,255,0),surface_identifier)
