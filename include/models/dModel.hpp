
#include "..\..\include\models\sModel.hpp"

#pragma once

/** Rewritable, fixed size mesh model */
class DModel : public SModel
{
public:
    DModel(float *vertexData, int nVertices);
    void setVertexData(float *vertexData, int nVertices);
};
