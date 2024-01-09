import pygame

BLOCKWIDTH = 50
TICK_COUNT = 20
LEFT_OFFSET = 100
TOP_OFFSET = 50


def init_graphics():
    global S_WIDTH
    S_WIDTH = 1000
    global S_HEIGHT
    S_HEIGHT = 1000
    global NODE_RADIUS
    NODE_RADIUS = 5
    pygame.display.init()
    pygame.font.init()
    global NODE_FONT
    NODE_FONT = pygame.font.Font('freesansbold.ttf', 15)
    global TITLE_FONT
    TITLE_FONT = pygame.font.Font('freesansbold.ttf', 40)
    global screen
    screen = pygame.display.set_mode((S_WIDTH, S_HEIGHT))
    global saved_surfaces
    saved_surfaces = {}

"""
data: list of quadruple numeericals (finshtime, duration,jobNumber,taskNumber,resourceNumber)

"""

def draw_gantt(blocks,finishTime,resourceCount):

    widthScale = 50
    for block in blocks:
        left = LEFT_OFFSET+(block[0]-block[1])*widthScale
        top = TOP_OFFSET+BLOCKWIDTH*block[4]                                            
        rect = pygame.Rect(left,top,block[1]*widthScale,BLOCKWIDTH)
        pygame.draw.rect(screen,(0,255,0),rect)
        pygame.draw.rect(screen,(255,0,0),pygame.Rect(left+2,top+2,block[1]*widthScale-4,BLOCKWIDTH-4))
        text = NODE_FONT.render("T({},{})".format(block[2],block[3]), True, (255,255,255))
        textRect = text.get_rect()
        textRect.center = rect.center
        screen.blit(text,textRect)
        pygame.display.update()
    wait()
    screen.fill((0,0,0))

def wait():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                return
            if event.type == pygame.QUIT:
                pygame.quit()