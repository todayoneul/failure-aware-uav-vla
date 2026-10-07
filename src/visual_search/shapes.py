"""Flat-coloured meshes as binary glTF, for objects added to a scene at run time. No simulator, no torch.

glTF is Y-up and in metres; the simulator maps glTF Y to its vertical axis, so `height` below is
what stands up in the world. Every mesh is centred on its own origin.
"""
import json
import math
import struct

KINDS=('box','cylinder','cone','pyramid')


def _pack(triangles,color,name):
    positions=[];normals=[]
    for a,b,c in triangles:
        u=[b[i]-a[i] for i in range(3)];v=[c[i]-a[i] for i in range(3)]
        n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
        length=math.sqrt(sum(x*x for x in n)) or 1.
        positions+=[a,b,c];normals+=[[x/length for x in n]]*3
    points=b''.join(struct.pack('<3f',*p) for p in positions);faces=b''.join(struct.pack('<3f',*n) for n in normals)
    indices=b''.join(struct.pack('<I',i) for i in range(len(positions)));blob=points+faces+indices
    low=[min(p[i] for p in positions) for i in range(3)];high=[max(p[i] for p in positions) for i in range(3)]
    document={'asset':{'version':'2.0'},'scene':0,'scenes':[{'nodes':[0]}],'nodes':[{'mesh':0,'name':name}],
              'meshes':[{'name':name,'primitives':[{'attributes':{'POSITION':0,'NORMAL':1},'indices':2,'material':0}]}],
              'materials':[{'name':name+'_material','pbrMetallicRoughness':{'baseColorFactor':[*color,1.],'metallicFactor':0.,'roughnessFactor':.9}}],
              'buffers':[{'byteLength':len(blob)}],
              'bufferViews':[{'buffer':0,'byteOffset':0,'byteLength':len(points),'target':34962},
                             {'buffer':0,'byteOffset':len(points),'byteLength':len(faces),'target':34962},
                             {'buffer':0,'byteOffset':len(points)+len(faces),'byteLength':len(indices),'target':34963}],
              'accessors':[{'bufferView':0,'componentType':5126,'count':len(positions),'type':'VEC3','min':low,'max':high},
                           {'bufferView':1,'componentType':5126,'count':len(normals),'type':'VEC3'},
                           {'bufferView':2,'componentType':5125,'count':len(positions),'type':'SCALAR'}]}
    text=json.dumps(document).encode();text+=b' '*(-len(text)%4);blob+=b'\0'*(-len(blob)%4)
    return (struct.pack('<4sII',b'glTF',2,28+len(text)+len(blob))+struct.pack('<I4s',len(text),b'JSON')+text
            +struct.pack('<I4s',len(blob),b'BIN\0')+blob)


def _box(size_x,size_y,height):
    x,y,z=size_x/2,height/2,size_y/2
    c=[(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),(-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)]
    triangles=[]
    for q in ((4,5,6,7),(1,0,3,2),(5,1,2,6),(0,4,7,3),(7,6,2,3),(0,1,5,4)):
        triangles+=[(c[q[0]],c[q[1]],c[q[2]]),(c[q[0]],c[q[2]],c[q[3]])]
    return triangles


def _prism(radius,height,sides,top,turn=0.):
    ring=[(math.cos(turn+2*math.pi*i/sides),math.sin(turn+2*math.pi*i/sides)) for i in range(sides)];triangles=[]
    for i in range(sides):
        (ax,az),(bx,bz)=ring[i],ring[(i+1)%sides]
        low_a=(radius*ax,-height/2,radius*az);low_b=(radius*bx,-height/2,radius*bz)
        high_a=(top*ax,height/2,top*az);high_b=(top*bx,height/2,top*bz)
        if top==0:triangles.append((low_a,high_a,low_b))
        else:triangles+=[(low_a,high_a,high_b),(low_a,high_b,low_b),((0,height/2,0),high_b,high_a)]
        triangles.append(((0,-height/2,0),low_a,low_b))
    return triangles


def mesh(kind,size_m,color,name='shape'):
    """GLB bytes for one object. `size_m` is its bounding box: [x, y, height]."""
    size_x,size_y,height=size_m
    if kind=='box':triangles=_box(size_x,size_y,height)
    elif kind=='cylinder':triangles=_prism(size_x/2,height,24,size_x/2)
    elif kind=='cone':triangles=_prism(size_x/2,height,32,0)
    # Four sides, turned so that the base edges run along the axes and match the bounding box.
    elif kind=='pyramid':triangles=_prism(size_x/math.sqrt(2),height,4,0,math.pi/4)
    else:raise ValueError(f'Unknown shape: {kind!r}')
    return _pack(triangles,color,name)
